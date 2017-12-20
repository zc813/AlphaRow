# SimpleAI

## Objective

To implement a vanilla version of game AI players as described in AlphaGo Zero--[Mastering the Game of Go Without Human Knowledge](https://www.nature.com/articles/nature24270) in Python.

## Brief Description

As described in the paper, the project mainly consists of three parts: _Self-play, Evaluation, and Optimization_.

This project builds a simple convolutional neural network, which takes the current board state as input and outputs predicted policies and a value. The network is then used in Self-play and optimized with data generated from the Self-play process. The parameters are updated when the agent with new parameters wins in Evaluation.

The project is currently written for TicTacToe. But it can be easy to adapt it for other games. Please note that this project is currently incomplete. There are still some flaws and additional work has to be done.

## How to Run

Self-play, Evaluation, and Optimization run in different processes. Since self-play tends to be the slowest, it is supported to run in multiple processes (three by default). For details, run 'train.py --help' in Terminal.

## Requirements

Keras and Tensorflow
