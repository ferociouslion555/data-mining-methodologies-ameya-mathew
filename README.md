# Data Mining Methodologies Assignment

**Student:** Ameya Mathew  
**SJSU ID:** 019110936

This repository demonstrates one end-to-end data-mining project through each of three methodologies: CRISP-DM, SEMMA, and KDD. Each project uses a different dataset and contains a reproducible notebook, methodology report, critical review, and deployment or demonstration artifact.

## Projects

1. [`crisp-dm-nyc-taxi`](crisp-dm-nyc-taxi/) — NYC taxi trip-duration regression and FastAPI deployment prototype.
2. [`semma-student-performance`](semma-student-performance/) — student-performance classification using Sample, Explore, Modify, Model, and Assess.
3. [`kdd-online-retail`](kdd-online-retail/) — transaction cleaning, customer segmentation, and association-rule discovery.

Complete notebooks are in [`notebooks/`](notebooks/), Medium-style article drafts are in [`articles/`](articles/), and the recording guide is [`video_walkthrough_script.md`](video_walkthrough_script.md).

## Running the notebooks

The notebooks are designed for Google Colab. Open the notebook and choose **Open in Colab**, or run locally with Python 3.11 and the packages in `requirements.txt`.

Each notebook downloads or explains the required dataset. Dataset files are intentionally not committed when their licenses or size make redistribution inappropriate.

## AI-assisted development and verification

Generative AI was used for brainstorming, code drafting, and adversarial critique. Code was reviewed manually, leakage controls were checked, metrics were recomputed, and claims were limited to the evidence actually generated. The critique logs in each project record unsupported claims and remaining limitations.

## Methodology comparison

| Methodology | Main emphasis | Project output |
|---|---|---|
| CRISP-DM | Business-to-deployment lifecycle | ETA regression and API prototype |
| SEMMA | SAS-style analytical modeling cycle | Student-performance classifier |
| KDD | Knowledge discovery from prepared data | Segments and retail rules |

## Submission website

The repository landing page is [`index.html`](index.html). Enable GitHub Pages from **Settings → Pages → Deploy from branch → main** and submit the resulting website URL.
