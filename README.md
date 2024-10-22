# Abstract

Autonomous mobile robots deployed to explore obstructed and unknown areas are required to perform Simultaneous Localization And Mapping (SLAM), and viewpoint planning (active SLAM), which involves determining the next action, whether aiming for a target point or following a path within a specific time horizon.

For an exploration mission, the agents aim to maximize the accuracy of their mapping while minimizing the time and distance traveled to cover the entire area.

In a multi-agent scenario, this challenge requires the agents to be aware of the opportunities related to their situation at each decision step to cooperate effectively.

Initial literature studied the problem of active SLAM for exploration without addressing localization issues. Subsequent work tackled localization errors using loop closure techniques and proposed revisiting actions. The multi-agent problem is often addressed through centralized and distributed systems, where communication between agents is possible and necessary for coordinated decision-making. In contrast, this work proposes to explore fully decentralized solutions, with individual mapping and self-planning. This approach addresses the active SLAM problem in a wide spectrum of situations, considering the potential imperfections in sensor accuracy and the constraint of limited communication between agents.

The exploration-exploitation dilemma is well-known in the single-agent scenario. In the multi-agent scenario, we extend it to a new stage involving exploring new areas, revisiting past locations to close loops, and searching for other agents to share map data and improve localization through multi-agent loop closure.

This work develops planners using expected reward maps called "cost-maps" and planners using Rapidly-exploring Random Trees (RRT) graphs and expected information gains based on possible future configurations of the environment. We analyze the behavior and performance of the agents for each planner in various exploration scenarios simulated with a discrete environment model. This work examines the benefits of long-term planning and estimating the position of other agents to increase the chance of encountering them. These improvements, such as relying on encountering other agents rather than backtracking to relocate, could allow for broader and faster exploration
