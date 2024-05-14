import numpy as np
import imageio
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Initialize the segmentation model
sam = sam_model_registry["vit_h"](checkpoint="D:\\Users\\MFE\\Xavier\\SAM\\sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Load a 3D image
image_3d = imageio.volread('D:\\Users\\MFE\\Xavier\\8IFnew\\image1_processed.tif')

# Container for segmented slices
segmented_slices = []

# Iterate over each slice of the 3D image
for i in range(5):
    # Extract the slice
    slice_2d = image_3d[i+30, :, :]

    # Convert to a PyTorch tensor, adding a channel dimension if grayscale
    slice_2d_torch = torch.from_numpy(slice_2d).unsqueeze(0).float()  # Channel first

    # Make sure the tensor is in the shape (C, H, W)
    if slice_2d_torch.shape[0] == 1:  # If only one channel
        slice_2d_torch = slice_2d_torch.expand(3, -1, -1)  # Expand to simulate RGB

    # Convert to PIL Image
    pil_image = to_pil_image(slice_2d_torch)

    # Convert PIL Image back to a NumPy array if necessary
    np_image = np.array(pil_image)

    print(f'Slice {i} shape: {np_image.shape}')

    # Perform the segmentation on the slice
    masks = mask_generator.generate(np_image)  # Ensure np_image is the expected type
    
    # Store the result
    segmented_slices.append(masks)

# Convert the list of segmented slices into a 3D numpy array
segmented_3d = np.stack(segmented_slices)

# Save the segmented 3D image
imageio.volwrite('D:\\Users\\MFE\\Xavier\\8IFnew\\SAM_pred.tif', segmented_3d)

print("Segmentation completed and saved.")
