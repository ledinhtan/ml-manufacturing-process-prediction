# Data

The datasets used in this project are **not publicly available** due to confidentiality agreements.  

This repository does **not include the original experimental data** used to train or evaluate the machine learning models.

---

## Data description

The data originate from manufacturing experiments related to **solid oxide cell (SOC) support fabrication**.  
Typical input variables include:

- `tape_id` – tape id   
- `temperature` – process temperature  
- `doctor_blade` – doctor blade gap  
- `casting_speed` – casting speed   
- `humidity` – relative humidity inside the tape casting machine 
- `volume_flow_rate` – Exhaust air volume flow rate of tape casting machine

Outputs correspond to measured properties of the tapes at different stages, for example:

- Green tape properties  
- Sintered tape properties  
- Reduced tape properties  

These data were generated within a collaborative research project and cannot be redistributed.

---

## Expected data format

If you wish to reuse the code with your own dataset, the pipeline expects **tabular data**:

- Rows correspond to experimental samples  
- Columns correspond to input parameters and measured outputs  

Example:

tape_id, temperature, doctor_blade, casting_speed, humidity, volume_flow_rate, green_thickness, green_density, sintered_thickness, sintered_density, reduced_thickness, reduced_density

1, 20, 200, 5, 50, 70, 66, 3.45, 63, 4.25, 62, 3.82 

2, 25, 200, 5, 30, 60, 65, 3.42, 63, 4.2, 62, 3.8 

...

---

## Folder structure

```bash
data/
├── raw/ # original experimental data (not included)
└── processed/ # cleaned data used for modelling (not included)
```

- Place your dataset in `data/processed/` to run the scripts or notebooks.  
- Ensure column names match those expected in the code.