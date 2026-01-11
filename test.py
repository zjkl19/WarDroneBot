import ultralytics, sysconfig
print("ultralytics module:", ultralytics.__file__)
print("Scripts dir:", sysconfig.get_paths()["scripts"])