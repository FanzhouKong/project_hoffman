"""Entry point of the standalone app (PyInstaller). freeze_support() must run first: inside the frozen app,
multiprocessing starts its workers by running this same program."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from peak3d.desktop import main
    main()
