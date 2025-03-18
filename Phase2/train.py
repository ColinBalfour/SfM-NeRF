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
print(device)
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
        

# def render(model, rays_origin, rays_direction, args, near=1.0, far=10.0):
#     """
#     Input:
#         model: NeRF model
#         rays_origin: origins of input rays
#         rays_direction: direction of input rays
#     Outputs:
#         rgb values of input rays
#     """
#     # print(rays_direction.shape, rays_origin.shape) # (N, 3) (N, 3)
#
#     # sample points along ray
#     # t_vals = torch.linspace(0, 1, args.n_sample)
#     # t_vals = near + (far - near) * t_vals
#     # t_vals = t_vals.unsqueeze(0).repeat(rays_origin.shape[0], 1).to(device)
#
#     # first, split into N bins, then uniformly randomly sample points in each bin to get t_vals
#
#     # this impl is bad for some reason
#     idx = torch.arange(args.n_sample, dtype=torch.float32).unsqueeze(0).to(device)
#     t_vals = near + (far - near) * (idx + torch.rand(rays_origin.shape[0], args.n_sample).to(device)) / args.n_sample
#
#     # this impl is slow
#     # t_vals = [] # (N, n_sample)
#     # for n in range(rays_origin.shape[0]):
#     #     tmp = []
#     #     for i in range(args.n_sample):
#     #         tmp.append(torch.uniform(near + i/args.n_sample * (far - near), near + (i+1)/args.n_sample * (far - near)))
#     #     t_vals.append(tmp)
#     # t_vals = torch.tensor(t_vals).to(device)
#
#
#     # get the delta for every sample
#     delta_t = t_vals[:, 1:] - t_vals[:, :-1]
#     delta_t = torch.cat([delta_t, torch.ones(delta_t.shape[0], 1).to(device)], dim=-1)
#
#     # get 3d coords
#     rays_direction = rays_direction.unsqueeze(1).repeat(1, args.n_sample, 1)
#     rays_origin = rays_origin.unsqueeze(1).repeat(1, args.n_sample, 1)
#     # print(rays_direction.shape, rays_origin.shape, t_vals.shape) # (N, n_sample, 3) (N, n_sample, 3) (N, n_sample)
#     rays_points = rays_origin + rays_direction * t_vals.unsqueeze(-1)
#
#     densities, rgbs = model(rays_points, rays_direction)
#
#     # print(densities.shape, rgbs.shape, delta_t.shape) # (N, n_sample, 1) (N, n_sample, 3) (N, n_sample)
#
#     alphas = 1.0 - torch.exp(-densities * delta_t.unsqueeze(-1))
#     weights = alphas * torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
#
#     # print(weights.shape) # (N, n_sample, 1)
#
#     prediction = torch.sum(weights * rgbs, dim=1)
#
#     # print(prediction.device)
#
#     return prediction

def render(model, rays_origin, rays_direction, args, near=2.0, far=6.0):
    """
    Render rays by sampling points and applying volume rendering
    Args:
        model: NeRF model
        rays_origin: ray origins, shape [N, 3]
        rays_direction: ray directions, shape [N, 3]
        args: arguments with sampling parameters
        near: near bound for sampling
        far: far bound for sampling
    Returns:
        rgb: rendered colors, shape [N, 3]
    """

    device = rays_origin.device
    batch_size = rays_origin.shape[0]

    # Stratified sampling of points along the ray
    t_vals = torch.linspace(0, 1, args.n_sample, device=device)
    t_vals = t_vals.expand(batch_size, args.n_sample)

    # Add randomness to sampling
    if args.perturb:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(t_vals.shape, device=device)
        t_vals = lower + (upper - lower) * t_rand

    # Convert to distances
    z_vals = near + (far - near) * t_vals

    # Prepare ray directions and origins for batch processing
    rays_o = rays_origin.unsqueeze(1).expand(-1, args.n_sample, -1)  # [N, n_samples, 3]
    rays_d = rays_direction.unsqueeze(1).expand(-1, args.n_sample, -1)  # [N, n_samples, 3]

    # Get sample positions
    pts = rays_o + rays_d * z_vals.unsqueeze(-1)  # [N, n_samples, 3]

    # Reshape for model prediction
    pts_flat = pts.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)

    # Run network (batched if needed)
    chunk_size = min(pts_flat.shape[0], args.chunk_size)
    densities = []
    rgbs = []

    for i in range(0, pts_flat.shape[0], chunk_size):
        pts_chunk = pts_flat[i:i + chunk_size]
        rays_d_chunk = rays_d_flat[i:i + chunk_size]

        density_chunk, rgb_chunk = model(pts_chunk, rays_d_chunk)
        densities.append(density_chunk)
        rgbs.append(rgb_chunk)

    # Combine chunks
    raw_density = torch.cat(densities, 0).reshape(batch_size, args.n_sample, 1)
    raw_rgb = torch.cat(rgbs, 0).reshape(batch_size, args.n_sample, 3)

    # Distance between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Add a small distance at the end
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], -1)

    # Volume rendering calculations
    # Alpha is 1 - exp(-density * distance)
    alpha = 1.0 - torch.exp(-raw_density.squeeze(-1) * dists)


    # Transmittance (probability of light reaching the point)
    # T = torch.ones_like(alpha)
    # for i in range(1, args.n_sample):
    #     T[:, i] = T[:, i - 1] * (1.0 - alpha[:, i - 1] + 1e-10)
    T = torch.ones_like(alpha)
    cumprod_term = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    T[:, 1:] = cumprod_term[:, :-1]

    # Weights = alpha * T
    weights = alpha * T

    # Render RGB
    rgb = torch.sum(weights.unsqueeze(-1) * raw_rgb, dim=1)

    return rgb


def loss(groundtruth, prediction):
    return nn.MSELoss()(groundtruth, prediction)

# def train(images, poses, camera_info, args):
#
#     model = NeRFmodel().to(device)
#     idx = 0
#
#     checkpoint_loaded = False
#     if args.load_checkpoint:
#         models = glob.glob(os.path.join(args.checkpoint_path, "model_*.pth"))
#         if len(models) == 0:
#             print("No checkpoint found... continuing from scratch")
#             if not os.path.exists(args.checkpoint_path):
#                 os.makedirs(args.checkpoint_path)
#         else:
#             def get_idx(str):
#                 return int(str.split("_")[-1].split(".")[0])
#
#             models = sorted(models, key=get_idx)
#
#             print("Loading checkpoint...")
#             model_pth = models[-1]
#             model.load_state_dict(torch.load(model_pth))
#             print(f"Checkpoint {model_pth} loaded")
#
#             idx = get_idx(model_pth)
#             print(f"Continue training from iteration {idx}")
#             checkpoint_loaded = True
#
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
#
#     logs = glob.glob(os.path.join("./logs", "*/"))
#     log_idx = 0
#     if len(logs) > 0:
#         # ['./logs/1/']
#         log_idx = max([int(log.split('/')[-2]) for log in logs])
#         if not checkpoint_loaded:
#             log_idx += 1
#
#     if args.log_id != "":
#         log_idx = args.log_id
#
#     log_pth = os.path.join(args.logs_path, f"{log_idx}")
#     writer = SummaryWriter(log_pth)
#     test_idx = [random.randint(0, len(images) - 1) for _ in range(5)]
#
#     try:
#         sum_loss = []
#         for i in tqdm(range(idx, args.max_iters)):
#             rays = generateBatch(images, poses, camera_info, args)
#             rays = torch.tensor(rays).to(device)
#
#             rays_origin = rays[:, :3]
#             rays_direction = rays[:, 3:6]
#             rays_rgb = rays[:, 6:]
#
#             prediction = render(model, rays_origin, rays_direction, args)
#
#             loss_value = loss(rays_rgb, prediction)
#             sum_loss.append(loss_value.item())
#
#             optimizer.zero_grad()
#             loss_value.backward()
#             optimizer.step()
#
#             if i % 100 == 0:
#                 writer.add_scalar('loss', loss_value.item(), i)
#                 writer.add_scalar('avg_loss', sum(sum_loss) / len(sum_loss), i)
#                 sum_loss = []
#
#             if i % 1000 == 0:
#                 # save images with tensorboard
#                 with torch.no_grad():
#                     model.eval()
#                     for j in range(5):
#                         idx = test_idx[j]
#                         image = images[idx]
#                         pose = poses[idx]
#                         prediction, loss_value = test_image(model, image, pose, camera_info, args)
#
#                         image = torch.tensor(image)
#                         pred_image = prediction.cpu().reshape(camera_info['height'], camera_info['width'], 3)
#
#                         image = image.permute(2, 0, 1) # Convert HxWxC -> CxHxW
#                         pred_image = pred_image.permute(2, 0, 1)  # Convert HxWxC -> CxHxW
#
#                         writer.add_image(f'gt_pred_image_{j}', torch.cat([image, pred_image], dim=2), i)
#                         writer.add_scalar(f'test_loss_{j}', loss_value, i)
#
#                     torch.cuda.empty_cache()
#                 model.train()
#
#
#             if i % args.save_ckpt_iter == 0:
#                 torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"model_{i}.pth"))
#
#     except KeyboardInterrupt:
#         print("Training interrupted, saving checkpoint...")
#     finally:
#         writer.close()
#         torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"model_{i}.pth"))
#         torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "final_model.pth"))
#
#     return
def train(images, poses, camera_info, args):
    """
    Train the NeRF model
    Args:
        images: training images
        poses: camera poses
        camera_info: camera parameters
        args: arguments
    """

    torch.autograd.set_detect_anomaly(True)

    # Initialize model
    model = NeRFmodel(
        embed_pos_L=args.n_pos_freq,
        embed_direction_L=args.n_dirc_freq,
        num_channels=args.hidden_size,
        pos_encoding=True
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay)

    # Initialize iteration counter
    start_iter = 0

    # Load checkpoint if requested
    if args.load_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pth")))

        if len(checkpoints) > 0:
            checkpoint_path = checkpoints[-1]
            print(f"Loading checkpoint from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iteration'] + 1

            print(f"Resuming from iteration {start_iter}")
        else:
            print("No checkpoint found, starting from scratch")

    # Setup tensorboard
    log_dir = os.path.join(args.logs_path, f"{args.exp_name}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Select random images for validation
    val_indices = np.random.choice(len(images), size=min(5, len(images)), replace=False)

    # Training loop
    pbar = tqdm(range(start_iter, args.max_iters))

    # Keep track of metrics
    running_loss = 0.0

    for iteration in pbar:
        model.train()

        # Generate batch of rays
        rays_batch = generateBatch(images, poses, camera_info, args)
        rays_batch = torch.tensor(rays_batch).to(device)

        # Split into ray origins, directions and target colors
        rays_o = rays_batch[:, :3]
        rays_d = rays_batch[:, 3:6]
        target_rgb = rays_batch[:, 6:9]

        # Forward pass
        optimizer.zero_grad()
        rendered_rgb = render(model, rays_o, rays_d, args, near=args.near, far=args.far)

        # Calculate loss
        loss = nn.MSELoss()(rendered_rgb, target_rgb)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update running loss
        running_loss += loss.item()

        # Update progress bar
        pbar.set_description(f"Loss: {loss.item():.6f}")

        # Log loss every 100 iterations
        if iteration % 100 == 0:
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iteration)

            avg_loss = running_loss / min(100, iteration - start_iter + 1)
            writer.add_scalar('Loss/train_avg', avg_loss, iteration)
            running_loss = 0.0

        # Validate and visualize every 1000 iterations
        if iteration % 1000 == 0:
            model.eval()

            with torch.no_grad():
                for i, idx in enumerate(val_indices):
                    val_img = images[idx]
                    val_pose = poses[idx]

                    # Render validation image
                    pred_img, val_loss = test_image(model, val_img, val_pose, camera_info, args)

                    # Log validation loss
                    writer.add_scalar(f'Loss/val_{i}', val_loss, iteration)

                    # Log images
                    # Convert to format expected by tensorboard
                    gt_img = torch.from_numpy(val_img).permute(2, 0, 1)  # HWC -> CHW
                    pred_img = pred_img.cpu().permute(2, 0, 1)  # HWC -> CHW

                    # Concatenate ground truth and prediction side by side
                    img_grid = torch.cat([gt_img, pred_img], dim=2)
                    writer.add_image(f'Val_Image_{i}', img_grid, iteration)

            # Flush writer to ensure logs are saved
            writer.flush()

            # Save checkpoint
            if iteration % args.save_ckpt_iter == 0:
                checkpoint_path = os.path.join(args.checkpoint_path, f"model_{iteration:06d}.pth")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_path, "final_model.pth")
    torch.save({
        'iteration': args.max_iters - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    # Close tensorboard writer
    writer.close()

    return model

# def test(images, poses, camera_info, args):
#
#     model = NeRFmodel().to(device)
#
#     if args.load_checkpoint:
#         model_pth = os.path.join(args.checkpoint_path, "final_model.pth")
#
#         if not os.path.exists(model_pth):
#             print("No final checkpoint found... loading latest checkpoint")
#
#             models = glob.glob(os.path.join(args.checkpoint_path, "model_*.pth"))
#             if len(models) == 0:
#                 print("No checkpoint found")
#                 return
#             else:
#                 print("Loading checkpoint...")
#                 model_pth = sorted(models)[-1]
#
#         model.load_state_dict(torch.load(model_pth))
#         print(f"Checkpoint {model_pth} loaded")
#
#     model.eval()
#
#     width = int(camera_info['width'])
#     height = int(camera_info['height'])
#
#     idx = random.randint(0, len(images) - 1)
#     image = images[idx]
#     pose = poses[idx]
#
#     prediction, loss_value = test_image(model, image, pose, camera_info, args)
#
#     pred_image = prediction.cpu().numpy().reshape(height, width, 3)
#
#     image = (image * 255).astype(np.uint8)
#     pred_image = (pred_image * 255).astype(np.uint8)
#
#     # display images
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Ground Truth")
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(pred_image)
#     plt.title("Prediction")
#     plt.axis('off')
#     plt.show()
#
#     return

def test(images, poses, camera_info, args):
    # Initialize model with the same architecture as used in training
    model = NeRFmodel(
        embed_pos_L=args.n_pos_freq,
        embed_direction_L=args.n_dirc_freq,
        num_channels=args.hidden_size,
        pos_encoding=True
    ).to(device)

    # Load checkpoint
    if args.load_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_path)
        final_model_path = os.path.join(checkpoint_dir, "final_model.pth")

        if os.path.exists(final_model_path):
            print(f"Loading final model from {final_model_path}")
            checkpoint = torch.load(final_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Find the latest checkpoint
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pth")))
            if len(checkpoints) == 0:
                print("No checkpoint found. Exiting.")
                return

            latest_checkpoint = checkpoints[-1]
            print(f"Loading latest checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.images_path, os.path.basename(args.checkpoint_path))
    os.makedirs(output_dir, exist_ok=True)

    # Get image dimensions
    width = int(camera_info['width'])
    height = int(camera_info['height'])

    # Test on multiple images (either specific ones or random ones)
    num_test_images = min(5, len(images))  # You can adjust this number
    test_indices = np.random.choice(len(images), size=num_test_images, replace=False)

    print(f"Testing on {num_test_images} random images...")

    total_loss = 0
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            print(f"Rendering test image {i + 1}/{num_test_images}...")

            # Get ground truth image and pose
            gt_image = images[idx]
            pose = poses[idx]

            # Render the image
            pred_image, loss_value = test_image(model, gt_image, pose, camera_info, args)
            total_loss += loss_value

            # Convert to numpy and scale to [0, 255]
            pred_np = pred_image.cpu().numpy()
            gt_np = gt_image

            pred_np_uint8 = (pred_np * 255).astype(np.uint8)
            gt_np_uint8 = (gt_np * 255).astype(np.uint8)

            # Save individual images
            imageio.imwrite(os.path.join(output_dir, f"test_{i}_gt.png"), gt_np_uint8)
            imageio.imwrite(os.path.join(output_dir, f"test_{i}_pred.png"), pred_np_uint8)

            # Create side-by-side comparison and save
            comparison = np.concatenate([gt_np_uint8, pred_np_uint8], axis=1)
            imageio.imwrite(os.path.join(output_dir, f"test_{i}_comparison.png"), comparison)

            # Display images
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(gt_np_uint8)
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(pred_np_uint8)
            plt.title(f"Prediction (Loss: {loss_value:.6f})")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"figure_{i}.png"), dpi=150)
            plt.show()

    avg_loss = total_loss / num_test_images
    print(f"Testing complete. Average loss: {avg_loss:.6f}")
    print(f"Results saved to {output_dir}")

# def test_image(model, image, pose, camera_info, args):
#
#     width = int(camera_info['width'])
#     height = int(camera_info['height'])
#
#     origins = []
#     directions = []
#     for y in range(height):
#         for x in range(width):
#             origin, direction = PixelToRay(camera_info, pose, (x, y), args)
#             origins.append(origin)
#             directions.append(direction)
#
#     origins = torch.tensor(np.array(origins, dtype=np.float32)).to(device)
#     directions = torch.tensor(np.array(directions, dtype=np.float32)).to(device)
#
#     # print(origins.shape, directions.shape) # (big) process only a batch of rays at a time
#
#     prediction = torch.zeros((height * width, 3)).to(device)
#     with torch.no_grad():
#         for i in tqdm(range(0, origins.shape[0], args.n_rays_batch)):
#             batch_origins = origins[i:i+args.n_rays_batch]
#             batch_directions = directions[i:i+args.n_rays_batch]
#
#             pred = render(model, batch_origins, batch_directions, args)
#             prediction[i:i+args.n_rays_batch] = pred
#
#     # print(prediction.shape, image.shape) # (height * width, 3) (height, width, 3)
#     loss_value = loss(torch.tensor(image.reshape(-1, 3)).to(device), prediction)
#
#     return prediction, loss_value.item()
#
def test_image(model, image, pose, camera_info, args):
    """
    Render a test image and compare with ground truth
    Args:
        model: NeRF model
        image: ground truth image
        pose: camera pose
        camera_info: camera parameters
        args: arguments
    Returns:
        prediction: rendered image
        loss_value: loss between rendered and ground truth image
    """
    device = next(model.parameters()).device
    width = int(camera_info['width'])
    height = int(camera_info['height'])

    # Generate all rays for the image
    origins = []
    directions = []

    for y in range(height):
        for x in range(width):
            origin, direction = PixelToRay(camera_info, pose, (x, y), args)
            origins.append(origin)
            directions.append(direction)

    origins = torch.tensor(np.array(origins, dtype=np.float32)).to(device)
    directions = torch.tensor(np.array(directions, dtype=np.float32)).to(device)

    # Process rays in batches
    prediction = torch.zeros((height * width, 3)).to(device)

    with torch.no_grad():
        for i in tqdm(range(0, origins.shape[0], args.n_rays_batch)):
            batch_origins = origins[i:i + args.n_rays_batch]
            batch_directions = directions[i:i + args.n_rays_batch]

            pred = render(model, batch_origins, batch_directions, args,
                          near=args.near, far=args.far)
            prediction[i:i + args.n_rays_batch] = pred

    # Calculate loss
    gt_image = torch.tensor(image.reshape(-1, 3)).to(device)
    loss_value = nn.MSELoss()(gt_image, prediction)

    # Reshape to image dimensions
    prediction_image = prediction.reshape(height, width, 3)

    return prediction_image, loss_value.item()

def main(args):
    # load data
    print("Loading data...")
    mode = args.mode
    #my mode = 'train'
    images, poses, camera_info = loadDataset(args.data_path, mode)
    
    args.n_rays_batch = int(args.n_rays_batch)
    args.n_sample = int(args.n_sample)
    args.max_iters = int(args.max_iters)
    args.n_pos_freq = int(args.n_pos_freq)
    args.n_dirc_freq = int(args.n_dirc_freq)
    args.perturb = bool(args.perturb)
    args.hidden_size = int(args.hidden_size)
    args.chunk_size = int(args.chunk_size)
    
    model_name = args.data_path.split("/")[-2]
    args.checkpoint_path = os.path.join(args.checkpoint_path, model_name)

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
    parser.add_argument('--n_rays_batch',default=32*32*8,help="number of rays per batch")
    parser.add_argument('--n_sample',default=256,help="number of sample per ray")
    parser.add_argument('--max_iters',default=200001,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--log_id',default="",help="log id")
    parser.add_argument('--exp_name', default="lego_experiment", help="experiment name for logging")
    parser.add_argument('--checkpoint_path',default="./Phase2/checkpoints_new_2/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--perturb', default= True, type=bool, help="use stratified sampling")
    parser.add_argument('--near', default=2.0, type=float, help="near bound for ray sampling")
    parser.add_argument('--far', default=6.0, type=float, help="far bound for ray sampling")
    parser.add_argument('--chunk_size', default=1024 * 32, type=int, help="chunk size for memory efficient inference")
    parser.add_argument('--hidden_size', default=256, type=int, help="hidden layer width")
    parser.add_argument('--lrate_decay', default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument('--lrate_decay_steps', default=50000, type=int, help="learning rate decay steps")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
