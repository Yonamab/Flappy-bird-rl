import sys

from flappybird_rl.runner import train, play, evaluate


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please choose a mode:")
        print("  python main.py train")
        print("  python main.py play")
        print("  python main.py evaluate")

    elif sys.argv[1] == "train":
        train()

    elif sys.argv[1] == "play":
        play()

    elif sys.argv[1] == "evaluate":
        evaluate()

    else:
        print("Unknown mode. Use train, play, or evaluate.")