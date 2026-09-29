from .config import load_config
from .data import prepare_data
from .model import ANN
from .utils import set_seed, plot_history
from .metrics import classification_report


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    d = prepare_data(cfg)

    m, t = cfg["model"], cfg["training"]
    model = ANN(d["X_train"].shape[1], m["hidden_layers"], m["hidden_activation"],
                m["dropout"], m["l2"])
    hist = model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                     epochs=t["epochs"], batch_size=t["batch_size"],
                     optimizer=t["optimizer"], lr=t["learning_rate"], patience=t["patience"])

    model.save(cfg["paths"]["model_path"])
    plot_history(hist, f"{cfg['paths']['figures_dir']}/training_curves.png")

    rep = classification_report(d["y_test"], model.predict(d["X_test"]))
    print(f"\nTest accuracy: {rep['accuracy']:.4f} | F1: {rep['f1']:.4f}")
    print(f"Model saved to {cfg['paths']['model_path']}")


if __name__ == "__main__":
    main()
