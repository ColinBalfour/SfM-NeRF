import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import cv2

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def  loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    
    images = []
    poses = []
    
    json_file = os.path.join(data_path, f"transforms_{mode}.json")
    
    print(f"Loading data from {json_file} for {mode}")
    with open(json_file, 'r') as f:
        data = json.load(f)
        camera_angle_x = data['camera_angle_x']
        frames = data['frames']
    
    for frame in frames:
        file_path = os.path.join(data_path, frame['file_path'] + ".png")
        # img = imageio.imread(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        if img.shape[-1] == 4:
            img = img[..., :3]
        
        images.append(img)
        poses.append(frame['transform_matrix'])
        
    focal = 0.5 * camera_angle_x * img.shape[1] / np.tan(0.5 * camera_angle_x)
    
    mtx = np.array([[focal, 0, img.shape[1] / 2],
                    [0, focal, img.shape[0] / 2],
                    [0, 0, 1]])

    camera_info = {
        'width': img.shape[1],
        'height': img.shape[0],
        'camera_matrix': mtx
    }

    return np.array(images), np.array(poses), camera_info
    

def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    
    pixel_x, pixel_y = pixelPosition
    width, height = camera_info['width'], camera_info['height']
    fx = camera_info['camera_matrix'][0][0]
    fy = camera_info['camera_matrix'][1][1]
    cx = camera_info['camera_matrix'][0][2]
    cy = camera_info['camera_matrix'][1][2]
    
    # get pixel position in camera frame
    pixel_x = (pixel_x - cx) / fx
    pixel_y = (pixel_y - cy) / fy
    pixel_z = 1.0
    
    pixel_position = np.array([pixel_x, pixel_y, pixel_z])
    
    # get ray direction in world frame
    ray_direction = np.dot(pose[:3, :3], pixel_position)
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # get ray origin in world frame
    ray_origin = pose[:3, 3]
    
    return ray_origin, ray_direction

def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """
    
    rays = []
    width = int(camera_info['width'])
    height = int(camera_info['height'])
    
    for i in range(args.n_rays_batch):
        image_index = random.randint(0, len(images) - 1)
        pixelPosition = (random.randint(0, width - 1), random.randint(0, height - 1))
        origin, direction = PixelToRay(camera_info, poses[image_index], pixelPosition, args)
        rgb = images[image_index][pixelPosition[1], pixelPosition[0]]
        # print([origin, direction, rgb])
        rays.append(np.concatenate([origin, direction, rgb], axis=0))
        
    return np.array(rays, dtype=np.float32)
        

def render(model, rays_origin, rays_direction, args, near=1.0, far=10.0):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    # print(rays_direction.shape, rays_origin.shape) # (N, 3) (N, 3)
    
    # sample points along ray
    t_vals = torch.linspace(0, 1, args.n_sample)
    t_vals = near + (far - near) * t_vals
    t_vals = t_vals.unsqueeze(0).repeat(rays_origin.shape[0], 1).to(device)
    
    # get the delta for every sample
    delta_t = t_vals[:, 1:] - t_vals[:, :-1]
    delta_t = torch.cat([delta_t, torch.ones(delta_t.shape[0], 1).to(device)], dim=-1)
    
    # get 3d coords
    rays_direction = rays_direction.unsqueeze(1).repeat(1, args.n_sample, 1)
    rays_origin = rays_origin.unsqueeze(1).repeat(1, args.n_sample, 1)
    # print(rays_direction.shape, rays_origin.shape, t_vals.shape) # (N, n_sample, 3) (N, n_sample, 3) (N, n_sample)
    rays_points = rays_origin + rays_direction * t_vals.unsqueeze(-1)
    
    densities, rgbs = model(rays_points, rays_direction)
    
    # print(densities.shape, rgbs.shape, delta_t.shape) # (N, n_sample, 1) (N, n_sample, 3) (N, n_sample)
    
    alphas = 1.0 - torch.exp(-densities * delta_t.unsqueeze(-1))
    weights = alphas * torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
    
    # print(weights.shape) # (N, n_sample, 1)
        
    prediction = torch.sum(weights * rgbs, dim=1)
    
    # print(prediction.device)
    
    return prediction

def loss(groundtruth, prediction):
    return nn.MSELoss()(groundtruth, prediction)

def train(images, poses, camera_info, args):
    
    model = NeRFmodel(3, 3).to(device)
    idx = 0
    
    checkpoint_loaded = False
    if args.load_checkpoint:
        models = glob.glob(os.path.join(args.checkpoint_path, "model_*.pth"))
        if len(models) == 0:
            print("No checkpoint found... continuing from scratch")
        else:
            def get_idx(str):
                return int(str.split("_")[-1].split(".")[0])
            
            models = sorted(models, key=get_idx)
            
            print("Loading checkpoint...")
            model_pth = models[-1]
            model.load_state_dict(torch.load(model_pth))
            print(f"Checkpoint {model_pth} loaded")
            
            idx = get_idx(model_pth)
            print(f"Continue training from iteration {idx}")
            checkpoint_loaded = True
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    
    logs = glob.glob(os.path.join("./logs", "*/"))
    log_idx = 0
    if len(logs) > 0:
        # ['./logs/1/']
        log_idx = max([int(log.split('/')[-2]) for log in logs])
        if not checkpoint_loaded:
            log_idx += 1
    
    log_pth = os.path.join(args.logs_path, f"{log_idx}")
    writer = SummaryWriter(log_pth)
    
    try:
        sum_loss = []
        for i in tqdm(range(idx, args.max_iters)):
            rays = generateBatch(images, poses, camera_info, args)
            rays = torch.tensor(rays).to(device)
            
            rays_origin = rays[:, :3]
            rays_direction = rays[:, 3:6]
            rays_rgb = rays[:, 6:]
            
            prediction = render(model, rays_origin, rays_direction, args)
            
            loss_value = loss(rays_rgb, prediction)
            sum_loss.append(loss_value.item())
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            if i % 100 == 0:
                writer.add_scalar('loss', loss_value.item(), i)
                writer.add_scalar('avg_loss', sum(sum_loss) / len(sum_loss), i)
                sum_loss = []

            if i % args.save_ckpt_iter == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"model_{i}.pth"))
        
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
    finally:
        writer.close()
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"model_{i}.pth"))
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "final_model.pth"))
    
    return

def test(images, poses, camera_info, args):

    model = NeRFmodel(3, 3).to(device)
    
    if args.load_checkpoint:
        model_pth = os.path.join(args.checkpoint_path, "final_model.pth")
        if not os.path.exists(model_pth):
            print("No final checkpoint found... loading latest checkpoint")
        
            models = glob.glob(os.path.join(args.checkpoint_path, "model_*.pth"))
            if len(models) == 0:
                print("No checkpoint found")
                return
            else:
                print("Loading checkpoint...")
                model_pth = sorted(models)[-1]

        model.load_state_dict(torch.load(model_pth))
        print(f"Checkpoint {model_pth} loaded")
        
    model.eval()
    
    width = int(camera_info['width'])
    height = int(camera_info['height'])

    idx = random.randint(0, len(images) - 1)
    image = images[idx]
    pose = poses[idx]
    
    origins = []
    directions = []
    for y in range(height):
        for x in range(width):
            origin, direction = PixelToRay(camera_info, pose, (x, y), args)
            origins.append(origin)
            directions.append(direction)
    
    origins = torch.tensor(np.array(origins, dtype=np.float32)).to(device)
    directions = torch.tensor(np.array(directions, dtype=np.float32)).to(device)
    
    # print(origins.shape, directions.shape) # (big) process only a batch of rays at a time
    
    prediction = torch.zeros((height * width, 3)).to(device)
    with torch.no_grad():
        for i in tqdm(range(0, origins.shape[0], args.n_rays_batch)):
            batch_origins = origins[i:i+args.n_rays_batch]
            batch_directions = directions[i:i+args.n_rays_batch]
            
            pred = render(model, batch_origins, batch_directions, args)
            prediction[i:i+args.n_rays_batch] = pred
    
    # print(prediction.shape, image.shape) # (height * width, 3) (height, width, 3)
    loss_value = loss(torch.tensor(image.reshape(-1, 3)).to(device), prediction)
    
    pred_image = prediction.cpu().numpy().reshape(height, width, 3)
    
    image = (image * 255).astype(np.uint8)
    pred_image = (pred_image * 255).astype(np.uint8)
    
    # display images
    print(max(image.flatten()), min(image.flatten()))
    print(max(pred_image.flatten()), min(pred_image.flatten()))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_image)
    plt.title("Prediction")
    plt.axis('off')
    plt.show()
    
    return

def main(args):
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/nerf_synthetic/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=100,help="number of sample per ray")
    parser.add_argument('--max_iters',default=100000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)