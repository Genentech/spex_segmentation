# Documentation: File and Folder Structure

This document describes the general file and folder structure of the project, how Conda environments are utilized, and the standard practices for organizing executable scripts. **All executable scripts for stages must be named `app.py`** to maintain consistency.

## Project Structure

- **Project root**
    - `manifest.json` — The main project manifest (if used). It may contain global settings or link individual pipeline stages.
    - Other files unrelated to a specific stage.

- **Stage folders** (e.g., `load_anndata`, `clustering`, `dimensionality_reduction`)
    - **`manifest.json`** (inside the stage folder)
        - Describes the key parameters required by this stage.
        - Contains the stage name, description, execution order (`stage`), input parameters (`params`), and expected output (`return`).
        - Defines dependencies (`depends_and_script`, `depends_or_script`) and environment settings (`conda`, `libs`, `conda_pip`).
        - **Conda Environments**: If `conda` is specified, the system creates or uses a Conda environment with the requested Python version and installs the necessary libraries (`libs` via Conda, `conda_pip` via pip).
        - **Parameter Definitions**: The `params` section in `manifest.json` must include precise details for all possible parameter types to ensure smooth communication between the client and the script.
        - **Result Transfer**: The result of each script is passed to the next stage through the structured output format defined in `return`.

    - **`app.py`** (the executable script for this stage)
        - This file must always be named `app.py` to maintain consistency.
        - It contains the core logic: reading data, processing it, and returning results.
        - Typically includes a `run(**kwargs)` function that:
            1. **Imports necessary dependencies** (e.g., `scanpy`, `numpy`).
            2. **Reads parameters** from `kwargs` (e.g., file paths, method choices, metrics).
            3. **Executes core functions** (e.g., data loading, clustering, dimensionality reduction).
            4. **Returns results** in the format defined in `manifest.json`.

## Parameter Types and Manifest Definitions

Each stage's `manifest.json` should specify parameters under `params` using the following structure:

```json
"params": {
"parameter_name": {
"name": "Parameter Name",
"label": "User-friendly label",
"description": "Detailed description of the parameter",
"type": "TYPE",
"required": true,
"default": "default_value",
"enum": ["option1", "option2"],
"min": 0,
"max": 100
}
}
```

### Supported Parameter Types

#### Basic Types
| Type       | Description & Manifest Example |
|------------|--------------------------------|
| `string`   | A text field. `{ "type": "string" }` |
| `int`      | Integer input. `{ "type": "int", "min": 0, "max": 100 }` |
| `float`    | Floating-point number. `{ "type": "float", "min": 0.0, "max": 1.0 }` |
| `enum`     | A dropdown selection. `{ "type": "enum", "enum": ["option1", "option2"] }` |

#### File and Data Types
| Type       | Description & Manifest Example |
|------------|--------------------------------|
| `file`     | A file selector. `{ "type": "file" }` |
| `dataGrid` | A structured table/grid input. `{ "type": "dataGrid" }` |

#### Image and Channel Selection
| Type       | Description & Manifest Example |
|------------|--------------------------------|
| `omero`    | Image selection from OMERO. `{ "type": "omero" }` |
| `channel`  | A single channel selector. `{ "type": "channel" }` |
| `channels` | Multi-channel selector. `{ "type": "channels" }` |

#### Job and Process Selection
| Type       | Description & Manifest Example |
|------------|--------------------------------|
| `job_id`   | A job selector. `{ "type": "job_id" }` |
| `process_job_id` | A process job selector. `{ "type": "process_job_id" }` |

These types are mapped to their respective React components in the UI, ensuring proper handling on the client-side.

## Supported File Formats

### **1. Image and Microscopy Data Formats (OMERO)**
OMERO supports various image formats, excluding those with a time dimension (e.g., time-lapse TIFFs).

| Format        | Description |
|--------------|-------------|
| **TIFF (.tif, .tiff)** | Multi-channel, multi-dimensional image storage widely used in microscopy. |
| **OME-TIFF (.ome.tif, .ome.tiff)** | A standardized format supporting structured metadata and multiple channels (CXY or CYXZ). |

**Unsupported Formats:**
- **TIFF stacks with time dimension (TXYC or TXYZC)** → Not supported for direct OMERO ingestion in this workflow.

### **2. AnnData File Format (H5AD)**
H5AD is a format used for storing annotated multi-dimensional data, particularly in single-cell transcriptomics and spatial biology.

#### **H5AD File Structure**
- **Observations (Cells or Regions) (`adata.obs`)**
    - `fov`, `volume`, `min_x`, `max_x`, `min_y`, `max_y` — Metadata defining spatial boundaries and properties.

- **Variables (Genes or Features) (`adata.var`)**
    - Contains `n_vars` variables (e.g., genes), with no additional annotations.

- **Spatial Data (`adata.obsm`, `adata.uns`)**
    - Stores spatial coordinates and additional metadata.

#### **Manifest Definition for H5AD Files**
```json
{
  "params": {
    "adata": {
      "name": "AnnData File",
      "label": "Spatial transcriptomics dataset",
      "description": "H5AD file containing spatial gene expression data",
      "type": "file",
      "required": true
    }
  }
}
```

This ensures structured integration of microscopy data while avoiding conflicts with unsupported formats in OMERO.

