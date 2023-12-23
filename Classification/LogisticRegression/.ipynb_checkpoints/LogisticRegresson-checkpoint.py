{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "29ac5766",
   "metadata": {},
   "source": [
    "## Import the Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "11ef2a6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "baf8603f",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('Social_Network_Ads.csv')\n",
    "X = dataset.iloc[:, :-1].values\n",
    "y = dataset.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33c0946f",
   "metadata": {},
   "source": [
    "## Split the data into Train and Test sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3001cd5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e536677",
   "metadata": {},
   "source": [
    "## Do Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "c4e75bf8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.80480212  0.50496393]\n",
      " [-0.01254409 -0.5677824 ]\n",
      " [-0.30964085  0.1570462 ]\n",
      " [-0.80480212  0.27301877]\n",
      " [-0.30964085 -0.5677824 ]\n",
      " [-1.10189888 -1.43757673]\n",
      " [-0.70576986 -1.58254245]\n",
      " [-0.21060859  2.15757314]\n",
      " [-1.99318916 -0.04590581]\n",
      " [ 0.8787462  -0.77073441]\n",
      " [-0.80480212 -0.59677555]\n",
      " [-1.00286662 -0.42281668]\n",
      " [-0.11157634 -0.42281668]\n",
      " [ 0.08648817  0.21503249]\n",
      " [-1.79512465  0.47597078]\n",
      " [-0.60673761  1.37475825]\n",
      " [-0.11157634  0.21503249]\n",
      " [-1.89415691  0.44697764]\n",
      " [ 1.67100423  1.75166912]\n",
      " [-0.30964085 -1.37959044]\n",
      " [-0.30964085 -0.65476184]\n",
      " [ 0.8787462   2.15757314]\n",
      " [ 0.28455268 -0.53878926]\n",
      " [ 0.8787462   1.02684052]\n",
      " [-1.49802789 -1.20563157]\n",
      " [ 1.07681071  2.07059371]\n",
      " [-1.00286662  0.50496393]\n",
      " [-0.90383437  0.30201192]\n",
      " [-0.11157634 -0.21986468]\n",
      " [-0.60673761  0.47597078]\n",
      " [-1.6960924   0.53395707]\n",
      " [-0.11157634  0.27301877]\n",
      " [ 1.86906873 -0.27785096]\n",
      " [-0.11157634 -0.48080297]\n",
      " [-1.39899564 -0.33583725]\n",
      " [-1.99318916 -0.50979612]\n",
      " [-1.59706014  0.33100506]\n",
      " [-0.4086731  -0.77073441]\n",
      " [-0.70576986 -1.03167271]\n",
      " [ 1.07681071 -0.97368642]\n",
      " [-1.10189888  0.53395707]\n",
      " [ 0.28455268 -0.50979612]\n",
      " [-1.10189888  0.41798449]\n",
      " [-0.30964085 -1.43757673]\n",
      " [ 0.48261718  1.22979253]\n",
      " [-1.10189888 -0.33583725]\n",
      " [-0.11157634  0.30201192]\n",
      " [ 1.37390747  0.59194336]\n",
      " [-1.20093113 -1.14764529]\n",
      " [ 1.07681071  0.47597078]\n",
      " [ 1.86906873  1.51972397]\n",
      " [-0.4086731  -1.29261101]\n",
      " [-0.30964085 -0.3648304 ]\n",
      " [-0.4086731   1.31677196]\n",
      " [ 2.06713324  0.53395707]\n",
      " [ 0.68068169 -1.089659  ]\n",
      " [-0.90383437  0.38899135]\n",
      " [-1.20093113  0.30201192]\n",
      " [ 1.07681071 -1.20563157]\n",
      " [-1.49802789 -1.43757673]\n",
      " [-0.60673761 -1.49556302]\n",
      " [ 2.1661655  -0.79972756]\n",
      " [-1.89415691  0.18603934]\n",
      " [-0.21060859  0.85288166]\n",
      " [-1.89415691 -1.26361786]\n",
      " [ 2.1661655   0.38899135]\n",
      " [-1.39899564  0.56295021]\n",
      " [-1.10189888 -0.33583725]\n",
      " [ 0.18552042 -0.65476184]\n",
      " [ 0.38358493  0.01208048]\n",
      " [-0.60673761  2.331532  ]\n",
      " [-0.30964085  0.21503249]\n",
      " [-1.59706014 -0.19087153]\n",
      " [ 0.68068169 -1.37959044]\n",
      " [-1.10189888  0.56295021]\n",
      " [-1.99318916  0.35999821]\n",
      " [ 0.38358493  0.27301877]\n",
      " [ 0.18552042 -0.27785096]\n",
      " [ 1.47293972 -1.03167271]\n",
      " [ 0.8787462   1.08482681]\n",
      " [ 1.96810099  2.15757314]\n",
      " [ 2.06713324  0.38899135]\n",
      " [-1.39899564 -0.42281668]\n",
      " [-1.20093113 -1.00267957]\n",
      " [ 1.96810099 -0.91570013]\n",
      " [ 0.38358493  0.30201192]\n",
      " [ 0.18552042  0.1570462 ]\n",
      " [ 2.06713324  1.75166912]\n",
      " [ 0.77971394 -0.8287207 ]\n",
      " [ 0.28455268 -0.27785096]\n",
      " [ 0.38358493 -0.16187839]\n",
      " [-0.11157634  2.21555943]\n",
      " [-1.49802789 -0.62576869]\n",
      " [-1.29996338 -1.06066585]\n",
      " [-1.39899564  0.41798449]\n",
      " [-1.10189888  0.76590222]\n",
      " [-1.49802789 -0.19087153]\n",
      " [ 0.97777845 -1.06066585]\n",
      " [ 0.97777845  0.59194336]\n",
      " [ 0.38358493  0.99784738]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test = sc.transform(X_test)\n",
    "print(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a3a01fc",
   "metadata": {},
   "source": [
    "## Train the Logistic Regression Model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ec510ddb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(random_state=0)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(random_state=0)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "classifier = LogisticRegression(random_state = 0)\n",
    "classifier.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "8665c603",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    }
   ],
   "source": [
    "# Predict the result\n",
    "i = classifier.predict([[-0.80480212, 0.50496393]])\n",
    "print(i)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97f3602a",
   "metadata": {},
   "source": [
    "## Test the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "93197a95",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = classifier.predict(X_test)\n",
    "y_pred = y_pred.reshape(len(y_pred), 1)\n",
    "y_test = y_test.reshape(len(y_test), 1)\n",
    "# print(np.concatenate((y_pred, y_test), 1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ebf26c21",
   "metadata": {},
   "source": [
    "## Make the confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "8ae8b873",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[65  3]\n",
      " [ 8 24]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.89"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(cm)\n",
    "accuracy_score(y_test, y_pred)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
