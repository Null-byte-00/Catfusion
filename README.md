# Catfusion: Diffusion model for generating cat images
## Model overview 
Catfusion conststs of 2 main parts. the noise scheduuler for adding noise to the image and a U-Net for doing the Backward process
### Noise scheduler 
the noise scheduler works by adding a random noise to the image based on the time step
![noise](https://github.com/Null-byte-00/Catfusion/blob/main/noise.png?raw=true)
### the U-Net
the U-Net consists of 5 down and 5 up layers with 10 dense layers for time embeddings that are added in each layer to the output
## Training 
I trained the model on my personal GPU for 2 epochs
## Dataset
The dataset id made up of 7000 Images collected fom [Cat API](https://thecatapi.com/)
Because the quality of Images wasn't good enough, I manually selected them
## Results 
The results weren't very satisfying. this is as close as I could get to an Image of a Cat
![Output](https://github.com/Null-byte-00/Catfusion/blob/main/catfusion_output.png?raw=true)
![Output](https://github.com/Null-byte-00/Catfusion/blob/main/catfusion_output2.png?raw=true)<br>
## License
Catfusion is published under CPL (Cat Public License). You are allowed to copy, modify and redistribute this software as long as the software you're using it in is related to cats and is available to the public under the same license
