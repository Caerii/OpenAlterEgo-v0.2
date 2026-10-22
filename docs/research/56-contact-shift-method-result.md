# One-cell displacement search: method-check result

The [development protocol](55-contact-shift-robust-search.md) fixed a
five-shift topological replay and a worst-shift selector. The
[seven-person program choices](results/hyser-contact-shift-method07-v1-selection.json)
were pushed at `699d7ca` before query scoring. The
[score artifact](results/hyser-contact-shift-method07-v1-score.json)
binds to selection SHA-256
`ef25b6d9c1feb999ce562f477f3736f987ef5714b0f68a01d24a7b733cdd6f40`.

Day one uses the nominal sites. For each of five placement shifts,
day-two support and query use the simultaneously measured neighboring
sites; the support decoder receives one labeled trial per gesture with
weight 4. The table reports each person's mean query balanced accuracy
over the five shifts.

| Subject | Worst-shift selected | Nominal selected | Fixed eight-site | Robust minus nominal | Robust minus fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 06 | 0.736 | 0.801 | 0.801 | -0.065 | -0.065 |
| 07 | 0.773 | 0.769 | 0.749 | +0.004 | +0.024 |
| 08 | 0.692 | 0.692 | 0.656 | 0.000 | +0.036 |
| 17 | 0.709 | 0.661 | 0.697 | +0.048 | +0.012 |
| 18 | 0.808 | 0.780 | 0.709 | +0.028 | +0.099 |
| 19 | 0.829 | 0.844 | 0.800 | -0.015 | +0.029 |
| 20 | 0.778 | 0.622 | 0.645 | +0.155 | +0.133 |
| **Mean** | **0.761** | **0.738** | **0.723** | **+0.022** | **+0.038** |

The robust program's mean personwise worst-shift accuracy was 0.719,
versus 0.692 for nominal selection and 0.683 for the fixed layout.
At zero shift, the means were 0.763, 0.732, and 0.732 respectively.

The predeclared compelling-result rule required robust-minus-nominal
mean at least +0.02, robust-minus-fixed mean at least +0.05, gains
versus fixed for five of seven people, and no loss versus fixed below
-0.05. It met the first and third conditions, but failed the second
and fourth. Subject 06 lost 0.065 to fixed. The +0.022 mean advantage
over nominal selection is dominated by subject 20: excluding that
person, the six-person mean robust-minus-nominal difference is
approximately zero. The method does **not** show a reliable superior
program-selection rule.

The simulator component here is a topological channel mapping backed
by simultaneous measured array channels, which is stronger evidence
than synthetic waveform resemblance for this specific intervention.
It remains limited: translations are one full grid cell, not known
millimetre displacements; all four grids shift in the same direction;
contact impedance, analog AFE noise, skin deformation, and facial
anatomy are absent. The classwise support split is not globally
chronological, and participants/day-two data were exposed in earlier
Hyser work. This is a method-specific forearm stress test, not
independent wearable validation.

The next model should represent *continuous uncertain placement and
contact quality* and be judged by held-out design ranking and regret.
Before adding more search scale or expensive physics, obtain a data
source with physically meaningful placement coordinates and repeated
sessions; the current grid topology cannot identify a millimetre
tolerance curve.
