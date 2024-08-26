
# Satellite Downlink Beamforming Demo

We can form transmission beams by manipulating the phase of a precoding for every antenna-user-link.
Number of antennas and users can be varied.
Pre-calculated "AI solutions" exist for every scenario. Note: These are approximations, not the optimum solution.

The relevant file to start the GUI demo is `src/gui.py`.

![screenshot.png](reports/screenshot.png)

### How does it work?

- The upper window shows the power gain of the superimposed antenna emissions per user.
- For the leftmost user only, the inset window shows the phase of each individual antenna emission.
- The phases can be controlled by the sliders on the right hand side.
- At any position, constructive interference of the antennas will lead to high directional gain, while destructive interference leads to low gain.
- To maximize the sum rate, we want to maximize each user's power gain at their position, while minimizing each user's power gain at all other user's positions.
- The lower window shows the effect of the overlapping gains. High power gain and low inter-user interference lead to high sum rates.
- The number in the lower window shows the sum rate, ignoring values lower 0.

```
root
|   .gitignore            | .gitignore
|   README.md             | this file
|   requirements.txt      | project dependencies
|           
+---reports               | related material
+---src                   | code
|   +---config            |   configuration files
|   +---models            |   learning related
|   +---satellite_figures |   tikz code to generate figures
|   +---images            |   gui images

```