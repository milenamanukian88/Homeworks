# Bass Diffusion Model: Light Phone III

## Project Overview
This project uses the Bass Diffusion Model to forecast the adoption of the Light Phone III in the United States from 2025 to 2035.

Historical smartphone ownership data from Pew Research Center is used to estimate the diffusion parameters.

## Model Parameters
- Coefficient of innovation (p): 0.0893
- Coefficient of imitation (q): 0.1968
- Historical market potential (M): 91.92%

## Main Results
- Estimated U.S. market potential (base scenario): 120,466 adopters
- Predicted cumulative adoption by 2035: approximately 104,000 adopters
- Peak annual adoption: 2028

## Project Structure
- `data/`: Historical smartphone data and forecast datasets
- `img/`: Graphs and visualizations
- `report/`: Final PDF report
- `.ipynb`: Jupyter Notebook containing the analysis and code

## How to Run
1. Open the Jupyter Notebook in Google Colab.
2. Run all cells in order.
3. Review the model parameters, forecasts, and graphs.

## Data Sources
- TIME Best Inventions 2025
- Pew Research Center
- TechCrunch
- U.S. Census Bureau

The full reference list is included in the notebook and PDF report.
