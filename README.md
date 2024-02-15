This project demonstrates object segmentation using the SAM (Segment Anything with Masking) model. SAM is a vision transformer-based model that can segment objects in images given a textual description.

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
    git clone https://github.com/your-username/your-repository.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the script `textSam.py` with the following command:

    ```bash
    python textSam.py <path to image> <'s'/'a'>
    ```

    - `<path to image>`: The path to the image file you want to segment.
    - `<'s'/'a'>`: Choose between 's' to view one mask at a time or 'a' to view all masks in one image.

2. Follow the prompts to enter the object you want to segment. Type 'exit' to stop the demo.

## Examples

- To segment a single object and view one mask at a time:

  ```bash
  python textSam.py path/to/image.jpg s
  ```

- To segment multiple objects and view all masks in one image:

  ```bash
  python textSam.py path/to/image.jpg a
  ```


## Acknowledgments

- The SAM model is based on the [OwlVision](https://arxiv.org/pdf/2205.06230.pdf) project by Google Research.
- The SAM model checkpoint used in this project is available at https://github.com/facebookresearch/segment-anything.
