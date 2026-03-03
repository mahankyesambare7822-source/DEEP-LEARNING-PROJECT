# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MAHANK YESAMBARE

*INTERN ID*: CTIS6742

*DOMAIN*: DATA SCIENCE

*MENTOR*: NEELA SANTOSH

# DISCRIPTION,


The Journey: Building a Real-Time AI Vision System in VS Code
1. The Foundation: Configuring the Digital Lab
Every great AI project starts with a clean workspace. I began by opening VS Code and creating a dedicated project folder named tain_ai. Because I am using Python 3.14—the absolute "bleeding edge" of the Python language—I couldn't just follow a standard tutorial. Most AI libraries aren't ready for a version this new, which meant I had to be the "lead engineer" of my own environment.

I initiated a Virtual Environment (venv) to ensure my system-wide Python stayed clean. In the VS Code terminal, I hit my first roadblock: Windows security policies blocked my scripts. I solved this by elevating my execution policy to RemoteSigned, a move that allowed my local scripts to run while keeping the system safe.

When I tried to install the famous TensorFlow, the terminal threw a "No matching distribution" error—Python 3.14 was just too new for it. Instead of giving up or downgrading, I pivoted to PyTorch. This was a strategic choice; PyTorch is highly flexible and allowed me to build deep learning models even on experimental Python versions.

2. Designing the "Brain": Architecture and Training
Once my environment was stable, I moved to the core of the project: The Neural Network. I created a script called train_314.py to build the AI's "intelligence." I chose the MNIST dataset, a legendary collection of 60,000 handwritten digits that serves as the gold standard for testing computer vision.

I designed a class called DigitBrain. In human terms, I was building a digital eye with 784 input nerves—one for every pixel in a 28x28 image. These "nerves" connected to a hidden layer of 128 neurons, which used a ReLU (Rectified Linear Unit) activation function to decide which visual patterns (like a curve or a straight line) were important. The final layer had 10 neurons, representing the digits 0 through 9.

I ran the training for 3 Epochs. During this phase, the AI was "studying." It looked at the images, guessed the number, calculated its "Loss" (how wrong it was), and used the Adam Optimizer to fix its internal weights. By the end, I saved this "frozen intelligence" into a file called brain_314.pth.

3. Creating the "Eyes": The Vision Pipeline
A brain is useless without a way to see the world. I wrote a second script, vision_314.py, to link my laptop’s webcam to the AI model. Using OpenCV, I captured a live video stream, but I didn't want the AI to be overwhelmed by my face or the room's background.

I engineered a Region of Interest (ROI)—a green bounding box in the center of the screen. This created a "focus zone." I then built a complex "vision pipeline" to translate reality into something the AI could understand. The high-definition, colorful camera frame was converted to grayscale, shrunk down to exactly 28x28 pixels, and inverted. This inversion was critical: since the AI was trained on white ink on a black background, I had to flip the colors of my black-ink-on-white-paper drawings so the AI wouldn't get confused.

4. Real-Time Inference and Hardware Safety
The final result was a high-speed feedback loop. Every few milliseconds, the code captured a frame, processed it, and fed it into the PyTorch model. I implemented Softmax logic to see not just what the AI thought it saw, but how confident it was.

One of the most professional touches I added was a try...finally block. In early tests, if I stopped the code, my laptop's camera light would stay on—a sign the hardware was still trapped. The finally block acted as a "Safety Shutdown," ensuring that whether I pressed 'q' to quit or the code crashed, the camera was released and the windows were destroyed properly.

5. Overcoming the "Version Gap"
The true value of this project lies in the technical troubleshooting. Because VS Code's Pylance engine initially struggled with the memory demands of Python 3.14, I had to manually adjust the Analysis Indexing Limits in the settings to prevent "Out of Memory" crashes.

I successfully bridged the gap between a brand-new programming language and complex mathematical libraries. I now have a functional, real-time AI that can be expanded to recognize faces, objects, or even sign language.

<img width="1916" height="1020" alt="Image" src="https://github.com/user-attachments/assets/e80e8425-836c-46eb-a5ce-f35ace1a1b00" />
