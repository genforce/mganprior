# Generative Models

First of all, we thank following repositories for their work on high-quality image synthesis

- [PGGAN](https://github.com/tkarras/progressive_growing_of_gans)
- [StyleGAN](https://github.com/NVlabs/stylegan)
- [StyleGAN2](https://github.com/NVlabs/stylegan2)

Compared to [InterFaceGAN](https://github.com/ShenYujun/InterFaceGAN) repo, this repo optimizes the common API for generator, improves the pipeline to convert tensorflow weights to pytorch version, and involves model for StyleGAN2.

Pre-trained tensorflow weights (either officially released or trained by ourselves) can be found from following links. Please download them and save to folder `pretrain/tensorflow/` before using.

**NOTE:** The officially released models are simply mirrored by us from the above three repositories, just in case they are not available from the official links.

| PGGAN Official | | | |
| :-- | :-- | :-- | :-- |
| Face
| [celebahq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERkthZuF1rBCrJRURQ5M1W8BbsfT5gFF-TGbuxCAuUJXPQ?e=uKYyQ1)
| Indoor Scene
| [bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZeWkI9pbUZDqZAzEUDjlSwB5nDZhe94vmmg4G5QSKGy7A?e=5RhTOo) | [livingroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbHv-4YvGYJJl6i4zH8s25kBqpA1RG-YZbAvp2PSc5CtRA?e=SnSk49) | [diningroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ee2LUJ6fectMiFDYYrZiA1sBD5q4j_FBC8xzH2Z6GSb-JQ?e=pxhVrt) | [kitchen-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERTDgXhOqJZPlM72bULyKsgBu7nABHvmCBIbwvASzKruvg?e=lIrB34)
| Outdoor Scene
| [churchoutdoor-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfPAIPVXbYxIn0KQ5IzCJxYBfEG4nP1p7D3MK-N24HLzow?e=za16Z1) | [tower-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXZGTFQX8gNPgwvCGWKmiIwBxGgU4UTIQy1wezKnpAADMg?e=KUp4hJ) | [bridge-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXba4rsRrcZDg_6SQk-vClMBmqesihPHY6fne5oobKLHhg?e=9Gk1v3)
| Other Scene
| [restaurant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb0vWXX-n5BLkf9jL61ekfwBxHDpFxVLq9igSYJyQ3x5FQ?e=EuqMTU) | [classroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW5vDIwjV6dPsfK_szdVnTABVsd11xvJ_O6-ReVeQsvtQA?e=dls0Jd) | [conferenceroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb1kh2L4ayxFjXQL2y34yEkBb_eZ9pcSXnY3ivnvCdeknA?e=wPATWN)
| Animal
| [person-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbILxVQAbd9HsjxXwiOX2PABWHvmIsgrdwmvF0PPQl8_Xw?e=799btl) | [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ebr89QFQnRJHv-OQ7IMgu-YBG02kswtRukk-9ylUqY8bGQ?e=ioo5m4) | [dog-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeC5DITcQUNFkBPaVFnS4-YBOpFaVb_5agq_vkPG_aFvlg?e=rnq8Rw) | [bird-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbvqTPl0ru5MicpQbuIePtgBSwDbzef23TgcrCNcFX5A-A?e=jMRaqB)
| [horse-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfsJ0u6ZhDhHvleYRd5OCYABCd6Q6uqU1l-AM_C-Cot5_g?e=Fqmudf) | [sheep-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaIy20hZi5pHkVZhO7p38OoBrjInx6UAFzwAMtG_fcnUCg?e=A6ax03) | [cow-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETcm1hzw7M5Mmbi1vHNAA1sBNZcCwXr1Y_y-nwVqEcNHKQ?e=IE0Cu0)
| Transportation
| [car-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ec6LMgv8jSpJo9MHv39boLkBR6zqrnK_XCjrJdDDoIjfTg?e=HKRIet) | [bicycle-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW9S1pnWUXtAuHLRcFeoHmYB0vmHwdf6ipxMIPOzxQnOaw?e=pbEBXp) | [motorbike-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESiSYhItLfZKnsuW1bA6XPMBHt9Um3p2WvEknOndLgNLtw?e=uVCCIx) | [bus-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EY996zZOBAFDip6W18m0OY0BERSAl_CoVJt0mCUNod2bBg?e=Mt8Qgg)
| [train-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeqoMneXJ6hKkuVoKTvgfG8Bbn7yx6FGByzzpF8avQ5ecw?e=7b0rb1) | [boat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESL-wYbgG2NMmEfjNqVe6DcB0wHkx-GeFsWWnmnhK6DL6Q?e=yVwAUW) | [airplane-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYsnUkaD7kZNjHLCeTcEEaYBjNPO6_wra4Erlh6SMCs3eQ?e=wvRubM)
| Furniture
| [bottle-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUjdW87xSmVCumxRS0E6OXUB67wxAjappdW4XHvbOx3UgA?e=GT46ho) | [chair-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYgI1WgBJ5NPomn9BedMkRkBTaKcQOIaoGWQg-oe-eVN8g?e=42YuAT) | [pottedplant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdxtDuYh_31Fpc6TA5ZtATQB2b2IqnwG0z4NzDzYfNHSOw?e=QV213z) | [tvmonitor-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfXmQpZbx35KuZkuZO_C_qsBYaFnnP6Cq9al4NI6-lrqLQ?e=Y2EEy8)
| [diningtable-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW0UuwPB3pZNh5jTUDHEl24BaIiqvcB-_9k1TpX3nRFhvw?e=xw7CYQ) | [sofa-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYHZdQA2DJBGjetN2agEnEEBicfxWbzMMON5wlgNDc5AFw?e=JsTzLG)

| StyleGAN Official | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| [ffhq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERrWZh5VmmFPkMPBqWyj88kBWevxE-_mELOo9toH8LEK9A?e=W7DS8j)     |    70,000 | 25,000 | 4.40
| [celebahq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZLLovAsvOBNhRrElHONwgYBsRy1QRc_kIOGpvHhEUar3w?e=ORRaR3) |    30,000 | 25,000 | 5.06
| [bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZZtgyznB4xNm9gmLyOBHpcB-ohvHMmKZmZx6Tfx9_o8HA?e=gV0ZXi)    | 3,033,042 | 70,000 | 2.65
| [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETHKOTaZPEBNkVoM-zJgDVQBciac5PZFsbVw8raYycDTlA?e=R8oxiP)        | 1,657,266 | 70,000 | 8.53
| [car-512x384](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EU-MbMEy8IlGtnjXv-dBF7cBZTJoGqG4le-NvyrURvS4Eg?e=IxDwat)        | 5,520,756 | 46,000 | 3.27

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

| StyleGAN2 Official | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| [ffhq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb053e-OAblEuS7ZnUto8R0Bub4HtnF5nJVoUUeLPA7Kbw?e=Kv3O8m) |    70,000 |  25,000 | 2.84
| [church-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ec88e85Iz_tMs9C2cSY1Bw4BCmVwCXJCFpHpWMkQ7HkcUA?e=ejGLi6) |   126,227 |  48,000 | 3.86
| [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERM88hE3pTlHm_8D9OMxwYUB20Yij8IshFwk0F6C2LV2pQ?e=4wmOQ5)    | 1,657,266 |  88,000 | 6.93
| [horse-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYdKeNQctlRHlFZX5aUIs2kBicFtS_MwSPSgemPJecVvzw?e=RiEM2H)  | 2,000,340 | 100,000 | 3.43
| [car-512x384](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWk22jukftBInK98BnW6hrgBSEPxtvYO4li8EFQhIj28wg?e=VeloK1)    | 5,520,756 |  57,000 | 2.32
