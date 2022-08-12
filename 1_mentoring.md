{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "1_mentoring.ipynb",
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "This is a matrix multiplication problem you are trying to solved. Only when the number of rows in the second matrix is equals to the number of columns in the first matrix can two matrices be multiplied. Therefore we need to change the shape of the second variable to mach with the first variable"
      ],
      "metadata": {
        "id": "kxeugJ8NJWKh"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 235
        },
        "id": "h_A4ufGqMdl0",
        "outputId": "9a4a0a62-4113-4524-fc9c-2134e0aeea20"
      },
      "source": [
        "import numpy as np\n",
        "\n",
        "a = np.array([[-1,2,3]])\n",
        "b = np.array([[0,2,1]])\n",
        "np.dot(a,b)"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-2-81fff3b7b83a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0ma\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<__array_function__ internals>\u001b[0m in \u001b[0;36mdot\u001b[0;34m(*args, **kwargs)\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: shapes (1,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0)"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "below is the code to check the shape of the variables in matrix"
      ],
      "metadata": {
        "id": "WEj_wTaqKsjA"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rLndWqJHON2J",
        "outputId": "8550b654-db89-4d92-e5b0-282a0b420f5a"
      },
      "source": [
        "a.ndim\n"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P5n6H7P5Ob4e",
        "outputId": "22ddd223-6e08-4efc-fd1b-ceea9c4d4b9a"
      },
      "source": [
        "b.ndim"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The two variables, a and b, both have the same dimension when you look at their shapes.\n",
        "In order to resolve the issue, you must modify the second matrix, which happens to be b, as seen below."
      ],
      "metadata": {
        "id": "i3IWVAAgLWI6"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "snexnpbVOg2s",
        "outputId": "52bda98b-5472-45b0-a037-426d3128c289"
      },
      "source": [
        "b = b.reshape(3, 1)\n",
        "b"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0],\n",
              "       [2],\n",
              "       [1]])"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "We will now run the code BELOW to examine the results after changing the form of the variable."
      ],
      "metadata": {
        "id": "r4V40FyV0Ln9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "np.dot(a,b)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rJQ7dsifyQmS",
        "outputId": "1c78f8e3-06ba-4328-fa49-03b79468e4ca"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[7]])"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    }
  ]
}