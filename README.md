This project demonstrates object segmentation using the SAM (Segment Anything with Masking) model. SAM is a vision transformer-based model that can segment objects in images given a proper prompt.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- Transformers
- OpenCV
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/SalvoSamba01/textual-image-segmentation
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download SAM model checkpoints at [SAM Github Repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)). 

## Usage (textSam.py)

1. Run the script `textSam.py` with the following command:

    ```bash
    python textSam.py <path to image> <'s'/'a'>
    ```

    - `<path to image>`: The path to the image file you want to segment.
    - `<'s'/'a'>`: Choose between 's' to view one mask at a time or 'a' to view all masks in one image.

2. Follow the prompts to enter the object you want to segment. Type 'exit' to stop the demo.


## Usage (SAM.py)

1. Run the script `SAM.py` with the following command:

    ```bash
    python textSam.py <path to image> <'l'/'h'/'b'>
    ```

    - `<path to image>`: The path to the image file you want to segment.
    - `<'l'/'h'/'b'>`: Choose between 'ViT-l', 'ViT-h' or 'ViT-b' model.
      
2. Follow the prompts.

## Examples

- To segment objects and view one mask at a time:

  ```bash
  python textSam.py path/to/image.jpg s
  ```

- To segment objects and view all masks in one image:

  ```bash
  python textSam.py path/to/image.jpg a
  ```

- To run SAM with ViT-h model:

  ```bash
  python SAM.py path/to/image.jpg h
  ```


## Acknowledgments

- The SAM model is based on the [OwlVision](https://arxiv.org/pdf/2205.06230.pdf) project by Google Research.
- The SAM model checkpoint used in this project is available at [SAM Github Repository](https://github.com/facebookresearch/segment-anything).
