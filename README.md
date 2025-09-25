# Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks  

**Noah Geiger<sup>1,2</sup>, Tamim Asfour<sup>1</sup>, Neville Hogan<sup>2,3</sup>, Johannes Lachner<sup>2,3</sup>**  

<sup>1</sup> KIT, Institute for Anthropomatics and Robotics, Germany  
<sup>2</sup> MIT, Department of Mechanical Engineering, USA  
<sup>3</sup> MIT, Department of Brain and Cognitive Sciences, USA  

🔗 [Project Website](https://strokeairobotics.github.io/DiffusionBasedImpedanceAdaptation)  
📄 [arXiv Paper](https://arxiv.org/abs/XXXX.XXXXX)  

---

## Abstract  

Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non-task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation.


## Environment Setup  

We recommend creating a conda environment using the provided `environment.yml` file.  

```bash
# Clone the repository
git clone https://github.com/StrokeAIRobotics/DiffusionBasedImpedanceAdaptation.git
cd DiffusionBasedImpedanceAdaptation

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ImpedanceLearning
```

## AVP Telemanipulation
The KUKA LBR iiwa can be **telemanipulated using the Apple Vision Pro** to collect your own training data.
Johannes: hardware


## AVP Telemanipulation (Interface)  

To telemanipulate the robot, the [TrackingStreamer app](https://github.com/Improbable-AI/VisionProTeleop) on the **Apple Vision Pro** must be started.  

Afterwards, update the IP address in:  
AVPTelemanipulation/avp_stream/VisionProCPPCommunication.py

Then run the script to start the interface communication between the **Apple Vision Pro** and the **C++ robot controller**:  

```bash
python AVPTelemanipulation/avp_stream/VisionProCPPCommunication.py
```
## AVP Telemanipulation (Robot Deployment) 
Johannes: robot deployment

## Data for Training and Inference Simulation
The model can be trained and evaluated on data collected via the telemanipulation procedure described above.  
Alternatively, the dataset provided in `Data/Parkour` can be used directly.  
This dataset was collected with the same teleoperation method and can serve as a starting point for both training and inference simulations.


## Impedance Learning

An example training and inference script is provided:

```bash
python ImpedanceLearning/DiffusionModel.py
```
This script demonstrates the complete workflow:
- Hyperparameters can be configured (e.g., number of noise-adding steps, beta start/end values, and many more).
- Training, validation, and test data are automatically loaded.
- The diffusion model is trained, evaluated, and an inference simulation is executed.
- Models and results are stored automatically.

This file is intended as an example of how to use the framework and can be extended or adapted for custom experiments.


## Robot deployment
hardware (iiwa type, force sensor)
build + debug + run (command in terminal)
Johannes
