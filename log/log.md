**2022-1-09**
conducted experiments on silhouette only, cannot produce image at all.

**2022-1-10**

Conducted several experiments on my synthesized dataset using cloud server. includes:
    - Textured bunny, trained for 45k steps, get psnr of ~ 30 seems cannot preserve high-freq info, may due to training. Beside, the disp is strange.
    - ![disp](img/Jan10Disp.png)  
    - Specular with outline, also 45k steps, the convergence  worse.   
    - ![train](img/Jan10Train.png)  

**2022-1-11**
For the specular with outline (silhouette only) it seems that it could some how encode the outline information.   
![outline](img/Jan11Outilne.png).   
However, it seems that it learns outline as view-dependent change of appearance. Notice the ear.
![outline2](img/Jan11Outline2.png)
<!-- todo: disable view-dependent for ablation -->

**2022-1-12**
configuration of server
read sections of occluding contours and neural contours,
maybe use a pix2pixHD for outline refinement

**2022-1-13**
read suggestive contours. lecture of geometry, differential geometry
experiment on occluding contours.

**2022-1-14**
experiment on occluding contours, trained for 200k steps.
the network learn majority of line to good quality, however some part is still not sharp. Artifacts are found on surface where there are Moire patterns.
![lowfreq](img/Jan14contour.png)
![art](img/Jan14Artifact.png)

todo: read BRDF, model the issue using sampling.