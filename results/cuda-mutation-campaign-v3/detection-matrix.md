# CUDA Mutation Detection Matrix

| Subject | Differential | Metamorphic | memcheck | racecheck | synccheck | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | detected, no_signal | detected, no_signal | no_signal | no_signal | no_signal | killed |
| R02 | detected | detected, no_signal | no_signal | no_signal | no_signal | killed |
| R03 | detected | no_signal | no_signal | no_signal | no_signal | killed |
| R04 | no_signal | no_signal | no_signal | detected | no_signal | killed |
| R05 | no_signal | no_signal | not_applicable | not_applicable | detected | killed |
| R06 | execution_error, no_signal | execution_error, no_signal | detected | no_signal | no_signal | killed |
| V01 | detected | detected | no_signal | no_signal | no_signal | killed |
| V02 | no_signal | no_signal | detected | no_signal | no_signal | killed |
| V03 | detected, no_signal | detected | no_signal | no_signal | no_signal | killed |
| V04 | detected | detected, no_signal | detected | no_signal | no_signal | killed |
| V05 | detected | detected, no_signal | no_signal | no_signal | no_signal | killed |
| V06 | detected | detected, no_signal | no_signal | no_signal | no_signal | killed |
| M01 | detected | detected, no_signal | no_signal | no_signal | no_signal | killed |
| M02 | detected, no_signal | detected, no_signal | no_signal | no_signal | no_signal | killed |
| M03 | no_signal | no_signal | detected | no_signal | no_signal | killed |
| M04 | detected | detected, no_signal | no_signal | no_signal | no_signal | killed |
| M05 | detected, no_signal | detected, no_signal | no_signal | detected | no_signal | killed |
| M06 | no_signal | no_signal | not_applicable | not_applicable | detected | killed |
