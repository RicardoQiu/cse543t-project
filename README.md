# cse543t-project
This repo is used for documenting all the files of the final project of cse543t at WashU
(Please feel free to make any change)

### Note
##### Please make sure all the code is included inside of the `code` folder. All the function or module should be located properly.

### Possible Thoughts
* Add regularizer to the original loss function;
* Analyze the convexity of loss function;
* using different optimizer to minimize the loss function; 



### Dataset
We will use the same dataset as used in the original paper

### Dependencies
Prior to running, adding all the following libraries via conda (feel free to add more):

* pytorch (1.5.1) and torchvision (0.6.0) 

  (GPU version) 
  ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```
  
  (CPU version) 
  ```conda install pytorch torchvision```
  
* scipy (1.4.1)
* matplotlib (3.2.2)
* numpy (1.18.1)
* skimage (0.17.2)
* tqdm (4.46.1) (visualize the ongoing process)

  ```conda install scipy matplotlib numpy scikit-image tqdm```

### Documentation
All the documentation is supposed to be hosted on Overleaf

Overleaf Link: https://www.overleaf.com/6326894856mfnjrztcmwft

### References

[comment]: <> (- [Image Style Transfer Using Convolutional Neural Networks]&#40;https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf&#41;)

[comment]: <> (- [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients]&#40;https://arxiv.org/pdf/2010.07468.pdf&#41;)
- [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf)

[comment]: <> (- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution]&#40;https://arxiv.org/pdf/1603.08155.pdf&#41;)

- ["A Neural Algorithm of Artistic Style"](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

- ["Image Style Transfer Using Convolutional Neural Networks"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

- ["Preserving Color in Neural Artistic Style Transfer"](https://arxiv.org/pdf/1606.05897.pdf) by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman

- ["Laplacian-Steered Neural Style Transfer"](https://arxiv.org/pdf/1707.01253.pdf) by Shaohua Li, Xinxing Xu, Liqiang Nie, Tat-Seng Chua

### Tutorials
- [Original Neural Style Transfer Algorithm (pytorch official--paper1)](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Fast Neural Style Transfer Algorithm (pytorch official--paper4)](https://github.com/pytorch/examples/tree/master/fast_neural_style)

