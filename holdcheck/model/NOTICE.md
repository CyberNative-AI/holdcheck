# SO-101 robot description

`so101.xml` is an unmodified copy of MuJoCo Menagerie `robotstudio_so101/so101.xml`, fetched 2026-11-15 from
`https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101/so101.xml`
(sha256 `982c02c027ec0c62d3459d0a6eb00dfb7dc13a451992e72ce99addc71f209805`). Apache License 2.0, text in `LICENSE`.
Menagerie derived it from TheRobotStudio/SO-ARM100 `Simulation/SO101/so101_new_calib.xml`.

`so101kin.py` reads only body frames, joint axes and ranges, inertial masses and centres, and the
`gripperframe` site. Meshes are not used and not copied.
