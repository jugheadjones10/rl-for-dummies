import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_csvs(experiment_name):
    all_dfs = []
    for dir in os.listdir("results"):
        if dir.startswith(experiment_name):
            file = os.path.join("results", dir, "results.csv")
            df = pd.read_csv(file)
            all_dfs.append(df)
    return all_dfs


# Create re-usable function for below plotting style:
def plot_seeded_results(name, title):
    # Create a new figure for the average plot
    plt.figure(figsize=(15, 6))

    dfs = parse_csvs(name)

    for df in dfs:
        plt.plot(df["step"], df["avg_score"])

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Score")
    plt.legend()
    plt.show()


def plot_average(name, title):
    plt.figure(figsize=(15, 6))

    dfs = parse_csvs(name)

    # Group by step and calculate mean
    combined_df = pd.concat(dfs)
    avg_df = combined_df.groupby("step").mean().reset_index()

    # Now step is a column again after reset_index()
    plt.plot(avg_df["step"], avg_df["avg_score"], "r-", linewidth=3)

    # Optional: Add standard deviation
    std_df = combined_df.groupby("step").std().reset_index()
    plt.fill_between(
        avg_df["step"],
        avg_df["avg_score"] - std_df["avg_score"],
        avg_df["avg_score"] + std_df["avg_score"],
        color="r",
        alpha=0.2,
        label="±1 std dev",
    )

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Score")
    plt.legend()
    plt.show()


def plot_multiple_averages(names, title):
    plt.figure(figsize=(15, 6))

    def plot_average(all_dfs, color, label):
        # Group by step and calculate mean
        combined_df = pd.concat(all_dfs)
        avg_df = combined_df.groupby("step").mean().reset_index()

        # Now step is a column again after reset_index()
        plt.plot(
            avg_df["step"], avg_df["avg_score"], color + "-", linewidth=3, label=label
        )

        # Optional: Add standard deviation
        # std_df = combined_df.groupby("step").std().reset_index()
        # plt.fill_between(
        #     avg_df["step"],
        #     avg_df["avg_score"] - std_df["avg_score"],
        #     avg_df["avg_score"] + std_df["avg_score"],
        #     color=color,
        #     alpha=0.2,
        #     label="±1 std dev",
        # )

    colors = ["r", "b", "g", "y", "m", "c", "k"]
    for name, color in zip(names, colors):
        dfs = parse_csvs(name)
        plot_average(dfs, color, name)

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Score")
    plt.legend()
    plt.show()
