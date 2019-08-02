# Image denoising

This work investigates a problem of denoising and describing images in the [N-HPatches dataset](https://github.com/hpatches/hpatches-dataset), a noisy adaptation of Homography patches (HPatches) dataset. The baseline approach explores a shallow U-Net in combination with L2-Net to process a noisy image to a set of descriptors. The improved approach consists of a deeper DNCNN with three denoising branches paired with a residual network to produce the descriptors. The best approach having 128 filters and 3, 5, 9 nodes per branch together with a 20-layer deep residual network achieved approximately 5%, 4% and 1% improvement over the baseline on patch retrieval, image matching and patch verification tasks respectively.

This project was a part of coursework for module EE3-25 Deep Learning at Imperial College London, [Link](http://intranet.ee.ic.ac.uk/electricalengineering/eecourses_t4/course_content.asp?c=EE3-25&s=E3#start).

## Structure
```
.
├── Baseline_approach.ipynb
├── Data
├── Figures
├── Improved_approach.ipynb
├── Models
├── README.md
├── Report.pdf
├── Report_interim.pdf
└── plot.py

```
The baseline is in `Baseline_approach.ipynb` and the improved approach is in `Improved_approach.ipynb`. The collected data is under `Data/` directory and all the Figures used in the report are in the `Figures/` directory. The pre-trained final models for baseline or improved approach are under the `Models/` directory. `plot.py` was used to plot the results for cross-validation and hyperparameter tuning, sourcing data directly from the `Data` directory.

All the outcomes are summarised in the [report](Report.pdf).

## Evaluation

In comparison to the baseline approach the improved approach achieved approximately 5%, 4% and 1% mean improvement on patch retrieval, image matching and patch verification tasks respectively. Overall, the improved approach achieved mean scores of 0.46 mAP on the patch re- trieval, 0.65 mAP on the patch verification and 0.12 mAP on the image matching. The training time increased approximately 20×, while the number of parameters for the descriptor network decreased, the number of parameters for the denoising network increased approximately 100×.

## Building & Running

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
jupyter notebook
# And then navigate to your desired notebook
```

and the figures will be found in `Figures/`.

## Credits
Martin Ferianc 2019.
