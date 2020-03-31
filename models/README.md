# Generative Models

First of all, we thank following repositories for their work on high-quality image synthesis

- [PGGAN](https://github.com/tkarras/progressive_growing_of_gans)
- [StyleGAN](https://github.com/NVlabs/stylegan)
- [StyleGAN2](https://github.com/NVlabs/stylegan2)

Compared to [InterFaceGAN](https://github.com/ShenYujun/InterFaceGAN) repo, this repo optimizes the common API for generator, improves the pipeline to convert tensorflow weights to pytorch version, and involves model for StyleGAN2.

Pre-trained tensorflow weights (either official released or trained by ourselves)can be found from following links. Please download them and save to path `models/pretrain/tensorflow` before using.

| PGGAN Official | | | |
| :-- | :-- | :-- | :-- |
| Face
| [celebahq-1024x1024](https://drive.google.com/a/google.com/file/d/188K19ucknC6wg1R6jbuPEhTq9zoufOx4/view?usp=sharing)
| Indoor Scene
| [bedroom-256x256](https://drive.google.com/a/google.com/file/d/1xbb2xakSn22lZoUcdaydQaBHoBSiUt6T/view?usp=sharing) | [livingroom-256x256](https://drive.google.com/a/google.com/file/d/1yhg7u-OqU8fYQ1A2GaMFwHg_bwKwmszF/view?usp=sharing) | [diningroom-256x256](https://drive.google.com/a/google.com/file/d/1yQRvnMsiMI5mksNkxtdnjJfZyY0W93i-/view?usp=sharing) | [kitchen-256x256](https://drive.google.com/a/google.com/file/d/1ycfRWFtiTl7EzELDQ9IArDnMAxBaGXcj/view?usp=sharing)
| Outdoor Scene
| [churchoutdoor-256x256](https://drive.google.com/a/google.com/file/d/1yGlooC5u4KuiOMzuJiA1W-MW4WB-wqnf/view?usp=sharing) | [tower-256x256](https://drive.google.com/a/google.com/file/d/1z8gUcWvUxtAjKhCzr88BO62NFBLYQ32G/view?usp=sharing) | [bridge-256x256](https://drive.google.com/a/google.com/file/d/1xf1SVhs52o93o_JhnPE2h6mZAM6YkI7t/view?usp=sharing)
| Other Scene
| [restaurant-256x256](https://drive.google.com/a/google.com/file/d/1yobti3l5kyeZxT-XJOAwnwRY7K_cdroY/view?usp=sharing) | [classroom-256x256](https://drive.google.com/a/google.com/file/d/1yJHhYysvxE4gVfYB56XVT_M4uea9DjpU/view?usp=sharing) | [conferenceroom-256x256](https://drive.google.com/a/google.com/file/d/1yLZ2YJ1ajh-amUJMpOmFHL9P3810xdqk/view?usp=sharing)
| Animal
| [person-256x256](https://drive.google.com/a/google.com/file/d/1ykf1q2wyOJufKUNGtgg2-jgcPvFA3NQd/view?usp=sharing) | [cat-256x256](https://drive.google.com/a/google.com/file/d/1xuFIDNAO_A_fVU0jFcgQd_C9A4Fn8GnT/view?usp=sharing) | [dog-256x256](https://drive.google.com/a/google.com/file/d/1yYmw3rAOIfOCZA0j8JvqYPEpe_NzM76A/view?usp=sharing) | [bird-256x256](https://drive.google.com/a/google.com/file/d/1xce6ct41eKxTjyASRsKY8erSikFpVWG5/view?usp=sharing)
| [horse-256x256](https://drive.google.com/a/google.com/file/d/1yaqsBv7e10svlLR_CuRyz7KB0TOjMWoq/view?usp=sharing) | [sheep-256x256](https://drive.google.com/a/google.com/file/d/1ysqX37xILHiVA0pscTwcg4FOabLfWQ5A/view?usp=sharing) | [cow-256x256](https://drive.google.com/a/google.com/file/d/1yMxQG3XNoQUBjTWqHvwJx7g2NyiFLNoH/view?usp=sharing)
| Transportation
| [car-256x256](https://drive.google.com/a/google.com/file/d/1xl6igEIL0N_wSLAocD6lyvTCsHkpRY88/view?usp=sharing) | [bicycle-256x256](https://drive.google.com/a/google.com/file/d/1xcMa9Tl6DXgnJD3CBnyBKd6Q4s_7w3kW/view?usp=sharing) | [motorbike-256x256](https://drive.google.com/a/google.com/file/d/1yj1FR6Oec-lCBLlNanVdXtG3VUNOYJXr/view?usp=sharing) | [bus-256x256](https://drive.google.com/a/google.com/file/d/1xkXOZa4RBdFvBsb-Ozn3iXEPER8d7W1B/view?usp=sharing)
| [train-256x256](https://drive.google.com/a/google.com/file/d/1zFmU835RjHGcelWkkc_0caIoxWJ4vpXQ/view?usp=sharing) | [boat-256x256](https://drive.google.com/a/google.com/file/d/1xco7Evj7XvCpUbZ6JA1d2yHYVpfu-Xei/view?usp=sharing) | [airplane-256x256](https://drive.google.com/a/google.com/file/d/18IA551HK_tuLETVm0E2RHnffCTu9UvD_/view?usp=sharing)
| Furniture
| [bottle-256x256](https://drive.google.com/a/google.com/file/d/1xeY_hVd7Z6_iw6-sJgz0bPIdxWBNSNYa/view?usp=sharing) | [chair-256x256](https://drive.google.com/a/google.com/file/d/1xxQnxpe5cr6RKtmfewtMR8R8O1KG_aFQ/view?usp=sharing) | [pottedplant-256x256](https://drive.google.com/a/google.com/file/d/1yo3Rtx_CJr83ZUxubw2uA_QnGmaBOofn/view?usp=sharing) | [tvmonitor-256x256](https://drive.google.com/a/google.com/file/d/1zHXSmE7320wzmxvBkwWZ_iHk8ArODhnn/view?usp=sharing)
| [diningtable-256x256](https://drive.google.com/a/google.com/file/d/1yPiqOi7xSiZTTctut4R2bP4ND4jqGMB8/view?usp=sharing) | [sofa-256x256](https://drive.google.com/a/google.com/file/d/1z4WTIl7TAHoLSUoapU4uxuHnouFWpAXZ/view?usp=sharing)

| StyleGAN Official |
| :-- |
| [ffhq-1024x1024](https://drive.google.com/a/google.com/file/d/1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ/view?usp=sharing)
| [celebahq-1024x1024](https://drive.google.com/a/google.com/file/d/1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf/view?usp=sharing)
| [bedroom-256x256](https://drive.google.com/a/google.com/file/d/1MOSKeGF0FJcivpBI7s63V9YHloUTORiF/view?usp=sharing)
| [cat-256x256](https://drive.google.com/a/google.com/file/d/1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ/view?usp=sharing)
| [car-512x384](https://drive.google.com/a/google.com/file/d/1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3/view?usp=sharing)

| StyleGAN Ours | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| **Face**
| [ffhq-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQBQjcqqgoNMkFNUjfE69soB5KPWHhSHW_LaOlH9WJ-uHw?e=AByHJO) |      70,000 | 25,000 |  5.70
| [ffhq-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUqBKsTPSX5KgBiN6mQ68fgB7pVdk4itrK7Budnxvd9FxA?e=wxVr6q) |      70,000 | 25,000 |  5.15
| **LSUN Indoor Scene**
| [livingroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW3M1ZzNc4REgBuFMD1soLgBQCteWBZdJsH7eCcRfJ-P-Q?e=LyfLj7) |   1,315,802 | 30,000 |  5.16
| [diningroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaZ1XWbU4KNKkD9SBUqtMXcBCq6ywjyeq-_OQ8sCUR6rzQ?e=rjOTcA) |     657,571 | 25,000 |  4.13
| [kitchen-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZ-3iOBSeqtKlIHWfOC4_-0BfzYwNHPNNYNVho2lkqm_Rw?e=TBAxAS) |   1,000,000 | 30,000 |  5.06
| **LSUN Indoor Scene Mixture**
| [apartment-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWvK04bleE1DrNO_GbtY4BsBZtqzSWJZ_VtxMkSJiK4QTg?e=WG74Jg) | 4 * 200,000 | 60,000 |  4.18
| **LSUN Outdoor Scene**
| [churchoutdoor-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcfDRkV7ncNJhJTsfbrli0MBnEPQXJeyNZ2FzS6XeAzKxA?e=Woibfx) |     126,227 | 30,000 |  4.82
| [tower-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXU65vZbVF5JhdqKWg8x7FkBXp8DCwdqPA26IkSiiKtLqw?e=nEkOQa) |     708,264 | 30,000 |  5.99
| [bridge-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWyzlQIgxNxOrcOkzb_GewkBqH5GTfKiMV1B27z5QJIJrw?e=6kgyan) |     818,687 | 25,000 |  6.42
| **LSUN Other Scene**
| [restaurant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZdxja8kJ8hFgVi4iCKApuoBRJ9HKUdNF53giR9D61V5jQ?e=8B1kLn) |     626,331 | 50,000 |  4.03
| [classroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfgXckBHSfZHsf_FUBXAsl8Btt6X0SRr1O8-FqyNbIaXRw?e=yZ5z8q) |     168,103 | 50,000 | 10.10
| [conferenceroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeOwgtZORopBibIOI022TYIBv1YPVpGy0FLM386olADZOg?e=hzaZzZ) |     229,069 | 50,000 |  6.20

| StyleGAN2 Official |
| :-- |
| [ffhq-1024x1024](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl)
| [church-256x256](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl)
| [cat-256x256](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl)
| [horse-256x256](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl)
| [car-512x384](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl)
