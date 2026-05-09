## Brief description 
Here I have an assignment. The detailed description is in `CandidateInstructions.md`. The model architecture can be found in `architecture.jpg` and I have translated it to `model_architecture.png` with the help of `weights.csv`, which contains more details of each layer. 


## Repository structure 
In this repo, I have initialized the python environment using uv and I have defined the repo structure as follows:
1. implement model architecture in `src/denoiser/denoise_model.py`. 
2. in `src/evaluation/compute_metrics.py`, compute PSNR and SSIM metrics between ground truth and predicted images, as well as ground truth with noisy images to check the improvement.
3. in `tests/test_inference.py`, implement a pytest on the image metrics. 
4. `main_inference.py` for inference pipeline. 
5. put inference and evaluation outputs in `ouputs/` folder, i.e., `denoised/` and `metrics/` respectively. 
6. `images/` contain images that need to be denoised and ground truth.

## Actions  
1. Convert the `weights.csv` to `checkpoint.pt` so that pytorch can load it more efficiently. The convertion is done only once, unless we want to update the weights. NOTE: All weights have been flattened from tensors with shape (in_channels, out_channels, kernel_height, kernel_width) in the CSV. Use pandas for all the CSV operations.
2. Double check the consistency between `architecture.jpg` and `model_architecture.png` with the help of `weights.csv`. Using pytorch to implement the model. The implementation MUST STRICTLY follow the architecture defined in `architecture.jpg`.
3. When implementing the inference, use torch.dataloader, enable both single image and batch inference.
4. Compute the PSNR and SSIM metrics between gt and predicted image, and compare with the metrics between gt and noisy image. Save the results in a csv file and plot the results using matplotlib, in order to show the perforance of the denoiser. 
5. Implent pytest using in `tests/test_inference.py`, compute the PSNR and SSIM between `image/gt/00.png` and the model predicted one (recompute here instead of using the one in `outputs`). The objective is to track the model performance while developing the code. 
6. Before writing the code, please make sure you have a clear understanding of what needs to be done and how it will be implemented, and show me the plan.

## General requirements  
1. I have added necessary packages in `pyproject.toml`, try to use these packages, do not use/install other packages if not necessary.
2. The code should be well-documented with comments explaining what each part of the code does, with Docstring in google style.
3. Your final code will be examinated by CLAUDE CODE. 

