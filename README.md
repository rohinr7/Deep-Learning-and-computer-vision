Here's a simplified `README.md`:

```markdown
# DLCV Lab Final - XAI Model Inference

**Author:** Rohin Andy Ramesh  
**Institution:** University of Bordeaux  

This project performs model inference using explainable AI (XAI) methods on test images, predicting classes and evaluating model explanations with metrics like PCC, SIM, deletion AUC, and insertion AUC. The script supports Grad-CAM, FEM, RICE, and LIME XAI methods.

## Setup

1. Clone the repository and navigate to the folder:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Initialize Git and commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Add DLCV Lab Final XAI project files"
   ```

## Usage

Run the script with the following command:

```bash
python main.py <modelpath> --testimages <optional> --saliencypath <optional> --mode <optional> --verbose <optional>
```

- `<modelpath>` is the path to the model file (required).
- Other parameters, like `--testimages`, `--saliencypath`, `--mode`, and `--verbose`, are optional.

### Examples

Run with default values:

```bash
python main.py Deep-Learning-and-computer-vision-main/resnet.pt
```

Specify custom test images:

```bash
python main.py Deep-Learning-and-computer-vision-main/resnet.pt --testimages /path/to/images
```

