import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config

sns.set_style("whitegrid")
sns.set_palette("deep")
mpl.rcParams["figure.figsize"] = config.DEFAULT_FIGSIZE
mpl.rcParams["lines.linewidth"] = config.DEFAULT_PLOT_LINEWIDTH
mpl.rcParams["lines.linestyle"] = config.DEFAULT_PLOT_LINESTYLE
mpl.rcParams["font.size"] = config.DEFAULT_AXIS_FONT_SIZE


def visualise_predictions(
    large_loans_dataset: str = "large_loans_gbm_prediction.parquet",
    small_loans_dataset: str = "small_loans_rf_prediction.parquet",
):

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger()

    large_loans = pd.read_parquet(config.FIN_FILE_PATH / large_loans_dataset)
    small_loans = pd.read_parquet(config.FIN_FILE_PATH / small_loans_dataset)

    for name, data in zip(["large_loans", "small_loans"], [large_loans, small_loans]):

        plt.clf()

        fig = plt.figure(figsize=(4, 10))
        gs = fig.add_gridspec(4, 1)
        gs.update(wspace=0.2, hspace=0.5)

        run = 0

        for row in range(4):
            for col in range(1):
                globals()[f"ax{run}"] = fig.add_subplot(gs[row, col])
                globals()[f"ax{run}"].set_yticklabels([])
                globals()[f"ax{run}"].tick_params(axis="y", which="both", length=0)
                for s in ["top", "right", "left"]:
                    globals()[f"ax{run}"].spines[s].set_visible(False)
                run += 1

        run = 0
        for col in ["loan_amount", "term", "income", "credit_score"]:
            sns.kdeplot(
                data=data,
                x=col,
                shade=True,
                ax=globals()[f"ax{run}"],
                linewidth=0.5,
                hue="prediction",
            )
            run += 1

        plt.savefig(config.REPORTS_PATH / f"{name}_continuous.jpeg")
        logger.info(f"VISUALISATION DONE {name}")


if __name__ == "__main__":
    visualise_predictions()

