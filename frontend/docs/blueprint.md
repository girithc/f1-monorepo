# **App Name**: Apex Edge

## Core Features:

- Health Check: Display the health status and model version from the /healthz endpoint.
- Circuit Metadata: Fetch and display circuit metadata, including overtake difficulty, from the /metadata endpoint.
- Single Scenario Prediction: Allow users to input race scenario details (circuit, grid position, pit plan, etc.) and display the predicted finish position distribution via the /predict endpoint.
- Strategy Comparison: Enable users to define and compare multiple race scenarios, displaying the recommended scenario based on the /compare endpoint.
- Sensitivity Analysis: Conduct a what-if analysis on starting grid positions to visualize finish position sensitivity using the /whatif endpoint.
- AI-Powered Pit Strategy Advisor: Suggest pit strategies, with lap and duration, to users aiming to optimize finish position using an LLM as a tool that analyzes car performance, tire score and circuit data.

## Style Guidelines:

- Primary color: Electric blue (#7DF9FF) to evoke a sense of speed, technology, and the futuristic aspects of racing strategy.
- Background color: Dark navy (#1A237E) providing contrast and a sophisticated backdrop for data visualization.
- Accent color: Neon green (#39FF14) for highlighting key metrics and interactive elements, drawing attention to important insights.
- Body and headline font: 'Space Grotesk', a sans-serif font that complements the tech-oriented feel of the app, suitable for both headlines and body text; for longer text use 'Inter' for body.
- Use minimalist, vector-based icons to represent different data points and actions. These icons should be easily understandable and align with the overall futuristic aesthetic.
- Employ a data-rich layout with clear sections for input parameters, prediction results, and comparative analyses. Use charts and graphs to visually represent data.
- Incorporate subtle animations for loading states and transitions, enhancing the user experience and providing feedback during calculations.