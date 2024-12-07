# GoalGuru for Borealis AI Let's Solve It
GoalGuru is a Slapshot analysis and classification pipeline created for the 2024 cohort of Borealis AI's Let's Solve It project internship. By leveraging cutting-edge pose estimation models and data-driven insights, GoalGuru analyzes slapshot mechanics to provide players with actionable feedback for improving shot speed, accuracy, and power efficiency.

# Key Features
- AI-Powered Shot Analysis: Uses Google MediaPipe to track body mechanics and extract essential biomechanical data from shot videos.
- Data-Driven Insights: Quantifies critical metrics like hand velocity, feet distance, and rotational sequence to refine technique.
- Custom Training Data: Built on high-quality shot clips collected with the University of Victoria hockey team.
- Advanced Modeling: Employs Long Short-Term Memory (LSTM) networks for precise shot classification and feedback.

# Future Goals
- Expand datasets across skill levels to enhance model robustness.
- Develop a mobile app for real-time shot recording, analysis, and progress tracking.
- Deliver advanced feedback for targeted improvements in player performance.

# Pipeline and File Organization Specifics
The pipeline works in order of these files:
1. pose_est.py: takes clipped_frames directory as input, runs pose estimation per frame, and outputs the coordinates as a json file.
2. stats.py: takes poses outputted from pose_est.py and computes stats which are then outputted as a json file.
3. log_reg.py: takes stats files as input, preprocesses into time series data (matrix of shape 70, 20, 8 for training, 40, 20, 8 for testing), and runs the data through a model of your choice.
  - NOTE: LSTM runs with highest accuracy.
