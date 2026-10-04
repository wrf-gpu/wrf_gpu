# W8 twin re-delivery on LW12 (6 twins 24 h) — run the minute LW12 is frozen
1. bash W8/setup.sh <LW12 commit>                      (CPU, detached snapshot, pin)
2. bash W8/launch.sh arm_a 24 20260227_18z_a1 20260502_18z_a1 20260614_18z_a1   (GPU lock, QUIET, cold compile in the first arm)
3. bash W8/launch.sh arm_b 24 20260220_18z_a1 20260608_18z_a1 20260120_18z_a1
4. bash W8/post_arm.sh W8/arm_a LW12 ; bash W8/post_arm.sh W8/arm_b LW12   (deflate -> D6+integrity -> manifest -> R32 formal)
5. python3 W8/deliver.py W8/arm_a LW12 ; python3 W8/deliver.py W8/arm_b LW12   (only cases with all 3 gates green)
