from .config import load_config
from .data import prepare_data
from .model import ANN
from .metrics import classification_report
from .utils import set_seed, plot_confusion_matrix


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    d = prepare_data(cfg)
    m = cfg["model"]
    model = ANN(d["X_test"].shape[1], m["hidden_layers"], m["hidden_activation"]).load(
        cfg["paths"]["model_path"])

    rep = classification_report(d["y_test"], model.predict(d["X_test"]))
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:>10}: {rep[k]:.4f}")
    plot_confusion_matrix(rep["confusion_matrix"],
                          save_path=f"{cfg['paths']['figures_dir']}/confusion_matrix.png")
    print("Confusion matrix saved.")


if __name__ == "__main__":
    main()
