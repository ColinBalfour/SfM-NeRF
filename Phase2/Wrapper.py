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
    
    # print(ray_origin)
    
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
    # t_vals = torch.linspace(0, 1, args.n_sample)
    # t_vals = near + (far - near) * t_vals
    # t_vals = t_vals.unsqueeze(0).repeat(rays_origin.shape[0], 1).to(device)
    
    # first, split into N bins, then uniformly randomly sample points in each bin to get t_vals
    
    # this impl is bad for some reason
    idx = torch.arange(args.n_sample, dtype=torch.float32).unsqueeze(0).to(device)
    t_vals = near + (far - near) * (idx + torch.rand(rays_origin.shape[0], args.n_sample).to(device)) / args.n_sample
    
    # this impl is slow
    # t_vals = [] # (N, n_sample)
    # for n in range(rays_origin.shape[0]):
    #     tmp = []
    #     for i in range(args.n_sample):
    #         tmp.append(torch.uniform(near + i/args.n_sample * (far - near), near + (i+1)/args.n_sample * (far - near)))
    #     t_vals.append(tmp)
    # t_vals = torch.tensor(t_vals).to(device)
    
    
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
    # total_weights = weights.sum(dim=[1, 2])
    # prediction = prediction + (1 - total_weights).unsqueeze(-1)
    
    # print(prediction.device)
    
    return prediction

def loss(groundtruth, prediction):
    return nn.MSELoss()(groundtruth, prediction)

def train(images, poses, camera_info, args):
    
    model = NeRFmodel().to(device)
    idx = 0
    
    checkpoint_loaded = False
    if args.load_checkpoint:
        models = glob.glob(os.path.join(args.checkpoint_path, "model_*.pth"))
        if len(models) == 0:
            print("No checkpoint found... continuing from scratch")
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
        else:
            def get_idx(str):
                return int(str.split("_")[-1].split(".")[0])
            
            models = sorted(models, key=get_idx)
            
            print("Loading checkpoint...")
            model_pth = models[-1]
            model.load_state_dict(torch.load(model_pth)['model_state_dict'])
            print(f"Checkpoint {model_pth} loaded")
            
            idx = get_idx(model_pth)
            print(f"Continue training from iteration {idx}")
            checkpoint_loaded = True
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    logs = glob.glob(os.path.join("./logs", "*/"))
    log_idx = 0
    if len(logs) > 0:
        # ['./logs/1/']
        log_idx = max([int(log.split('/')[-2]) for log in logs])
        if not checkpoint_loaded:
            log_idx += 1
            
    if args.log_id != "":
        log_idx = args.log_id
    
    log_pth = os.path.join(args.logs_path, f"{log_idx}")
    writer = SummaryWriter(log_pth)
    test_idx = [random.randint(0, len(images) - 1) for _ in range(5)]
    
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
            scheduler.step()
            
            if i % 100 == 0:
                writer.add_scalar('loss', loss_value.item(), i)
                writer.add_scalar('avg_loss', sum(sum_loss) / len(sum_loss), i)
                sum_loss = []
                
            if i % 1000 == 0:
                # save images with tensorboard
                with torch.no_grad():
                    model.eval()
                    for j in range(5):
                        idx = test_idx[j]
                        image = images[idx]
                        pose = poses[idx]
                        prediction, loss_value = test_image(model, image, pose, camera_info, args)
                        
                        image = torch.tensor(image)
                        pred_image = prediction.cpu().reshape(camera_info['height'], camera_info['width'], 3)
                        
                        image = image.permute(2, 0, 1) # Convert HxWxC -> CxHxW
                        pred_image = pred_image.permute(2, 0, 1)  # Convert HxWxC -> CxHxW
                        
                        writer.add_image(f'gt_pred_image_{j}', torch.cat([image, pred_image], dim=2), i)
                        writer.add_scalar(f'test_loss_{j}', loss_value, i)
                
                    torch.cuda.empty_cache()
                model.train()
                

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

    model = NeRFmodel().to(device)
    
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

        model.load_state_dict(torch.load(model_pth)['model_state_dict'])
        print(f"Checkpoint {model_pth} loaded")
        
    model.eval()
    
    width = int(camera_info['width'])
    height = int(camera_info['height'])

    idx = random.randint(0, len(images) - 1)
    image = images[idx]
    pose = poses[idx]
    
    prediction, loss_value = test_image(model, image, pose, camera_info, args)
    
    pred_image = prediction.cpu().numpy().reshape(height, width, 3)
    
    image = (image * 255).astype(np.uint8)
    pred_image = (pred_image * 255).astype(np.uint8)
    
    # display images
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_image)
    plt.title("Prediction")
    plt.axis('off')
    
    plt.savefig(os.path.join(args.images_path, f"pred_{idx}.png"))
    plt.show()
    

def test_image(model, image, pose, camera_info, args):
    
    width = int(camera_info['width'])
    height = int(camera_info['height'])
    
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
    
    return prediction, loss_value.item()

def test_single_image(images, poses, camera_info, args):
    """
    Testing regime for the NeRF model on a single image.

    Keyword Args:
        tn: Near bound. (default: {2})
        tf: Far bound. (default: {6})
        samples: Number of samples per ray. (default: {192})
        batch_size: Number of rays per batch. (default: {256})
        H: Height of the image. (default: {400})
        W: Width of the image. (default: {400})
        log_dir: Directory to save logs. (default: {'trial_logs'})

    Returns:
        The rendered RGB colors for the input ray.
    """
    # Camera parameters
    focal_length = camera_info['camera_matrix'][0][0]
    H = camera_info['height']
    W = camera_info['width']
    

    model = NeRFmodel().to(device)
    
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

        model.load_state_dict(torch.load(model_pth)['model_state_dict'])
        print(f"Checkpoint {model_pth} loaded")
        
    model.eval()
    
    pixel_values= []

    # Plot rays
    import matplotlib.pyplot as plt
    def plot_rays(o, d, t):
        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        # Label axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        pt1 = o
        pt2 = o + t * d

        for p1, p2 in zip(pt1[::50], pt2[::50]):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

        plt.show()

    trans_t = lambda t : np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=np.float32)

    rot_phi = lambda phi : np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
    ], dtype=np.float32)

    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
    ], dtype=np.float32)


    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w

    frames = []
    count = 0
    for th in tqdm(np.linspace(0., 360., 10, endpoint=False)):
        count += 1
        c2w = pose_spherical(th, -30., 4.)
        
        rays_origin = []
        rays_direction = []
        for y in range(H):
            for x in range(W):
                origin, direction = PixelToRay(camera_info, c2w, (x, y), args)
                rays_origin.append(origin)
                rays_direction.append(direction)
        
        rays_origin = np.array(rays_origin, dtype=np.float32)
        rays_direction = np.array(rays_direction, dtype=np.float32)
        all_rays = np.concatenate((rays_origin, rays_direction), axis=-1)

        # Plot rays
        plot_rays(rays_origin, rays_direction, 6)

        pixel_values = []
        for i in tqdm(range(0, len(all_rays), args.n_rays_batch)):
            batch = all_rays[i:i+args.n_rays_batch]

            rays_origin = torch.tensor(batch[:, :3]).to(device)
            rays_direction = torch.tensor(batch[:, 3:6]).to(device)

            predicted_pixel_values = render(model, rays_origin, rays_direction, args)
            pixel_values.append(predicted_pixel_values.detach().cpu())

        img = torch.cat(pixel_values).numpy().reshape(H, W, 3)*255.0
        frames.append(img)
        out_pth = os.path.join(args.images_path, f"image_{count}.png")
        cv2.imwrite(out_pth, img)

def main(args):
    # load data
    print("Loading data...")
    mode = args.mode
    mode = 'train'
    images, poses, camera_info = loadDataset(args.data_path, mode)
    
    args.n_rays_batch = int(args.n_rays_batch)
    args.n_sample = int(args.n_sample)
    args.max_iters = int(args.max_iters)
    args.n_pos_freq = int(args.n_pos_freq)
    args.n_dirc_freq = int(args.n_dirc_freq)
    
    model_name = args.data_path.split("/")[-2]
    args.checkpoint_path = os.path.join(args.checkpoint_path, model_name)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

    elif args.mode == 'gif':
        print("Start gif")
        args.load_checkpoint = True
        test_single_image(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/nerf_synthetic/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*8,help="number of rays per batch")
    parser.add_argument('--n_sample',default=256,help="number of sample per ray")
    parser.add_argument('--max_iters',default=100001,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--log_id',default="",help="log id")
    parser.add_argument('--checkpoint_path',default="./Phase2/checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)