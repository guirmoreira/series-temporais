import re
import stat
import warnings
from logging import config

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import config_context, metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings('ignore')


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


class TimeSeries:
    def __init__(self, path: str, title: str, ylabel: str, time_format: str = '%m-%Y'):
        self.path = path
        self.series = pd.read_csv(path)
        self.series['Date'] = pd.to_datetime(
            self.series['Time'], format=time_format)
        self.series['Year'] = self.series['Date'].dt.year
        self.series['Month'] = self.series['Date'].dt.month
        self.data = self.series['Value']
        self.title = title
        self.ylabel = ylabel

    def __repr__(self):
        return f"TimeSeries({self.path}, {self.title}, {self.ylabel})"

    def __str__(self):
        return f"TimeSeries({self.path}, {self.title}, {self.ylabel})"

    def __len__(self):
        return len(self.series)

    def plot_series(self):
        ax = self.series.plot(x='Date', y='Value',
                              color='blue', linestyle='-', marker='.')
        plt.gcf().set_size_inches(12, 4)
        plt.title(self.title)
        plt.xlabel("Data")
        plt.ylabel(self.ylabel)
        min_date = self.series['Date'].min()
        max_date = self.series['Date'].max()
        ax.set_xlim(min_date, max_date)
        min_year = min_date.year
        max_year = max_date.year
        tick_dates = pd.date_range(
            start=f"{min_year}-01-01", end=f"{max_year}-01-01", freq='YS')
        ax.set_xticks(tick_dates)
        ax.set_xticklabels([date.strftime('%Y') for date in tick_dates])
        plt.xticks(rotation=45)
        ax.grid(True, which='major', axis='both',
                linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def plot_hist(self, data=None):
        if data is None:
            data = self.data
        # 'bins' define o número de barras
        plt.hist(self.data, bins=20, edgecolor='blue')
        plt.gcf().set_size_inches(4, 3)
        plt.title('Histograma')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.text(0.58, 0.8, f'Mean: {np.mean(self.data):.4f}\nVar: {(np.std(self.data))**2:.2f}',
                 transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.show()

    def _config_acf_pacf_plot(self, series, func, title, lags):
        func(series, lags=lags)
        plt.gcf().set_size_inches(8, 3)
        plt.title(title)
        plt.xticks(np.arange(0, lags+2, step=2))
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelação')
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.gca().set_ylim(-0.5, 1.1)
        plt.show()

    def acf_plot(self, data, lags):
        self._config_acf_pacf_plot(data, plot_acf, f"ACF - {self.title}", lags)

    def pacf_plot(self, data, lags):
        self._config_acf_pacf_plot(
            data, plot_pacf, f"PACF - {self.title}", lags)

    def acf_pacf_plot(self, data=None, lags=30):
        if data is None:
            data = self.data
        self.acf_plot(data, lags)
        self.pacf_plot(data, lags)

    def adf_test(self, verbose=True):
        vprint("---\nTeste ADF:", verbose=verbose)
        result = adfuller(self.data)
        vprint(f"ADF Statistic: {result[0]}", verbose=verbose)
        vprint(f"p-value: {result[1]}", verbose=verbose)
        vprint("Conclusão:", "Série Estacionária" if result[1]
               < 0.05 else "Série Não Estacionária", verbose=verbose)
        return result[1] < 0.05

    def kpss_test(self, verbose=True):
        vprint("---\nTeste KPSS:", verbose=verbose)
        result = kpss(self.data, regression='c')
        vprint(f"KPSS Statistic: {result[0]}", verbose=verbose)
        vprint(f"p-value: {result[1]}", verbose=verbose)
        vprint(
            "Conclusão:", "Série Não Estacionária" if result[1] < 0.05 else "Série Estacionária", verbose=verbose)
        return result[1] >= 0.05

    def decompose(self, model='additive', freq="MS", period=12):

        # Criando uma série temporal com frequência mensal
        self.data = self.series.set_index('Date')['Value']
        self.data = self.data.dropna()
        _df = self.data.asfreq(freq=freq)

        # Decompondo a série temporal usando o modelo aditivo
        decomposition = sm.tsa.seasonal_decompose(
            _df, model=model, period=period)

        # Acessando os componentes da decomposição
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual

        # Mantendo a escala dos gráficos para comparação
        min_value = min(_df.min(), trend.min(), seasonal.min(), residual.min())
        max_value = max(_df.max(), trend.max(), seasonal.max(), residual.max())
        min_value = min_value-(max_value-min_value)*0.1
        max_value = max_value+(max_value-min_value)*0.1

        # Plotando os componentes
        plt.figure(figsize=(12, 10))

        years = mdates.YearLocator()   # every year
        years_fmt = mdates.DateFormatter('%Y')

        def config_plot(ax):
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(years_fmt)
            ax.xaxis.set_tick_params(rotation=30, labelsize=8)
            ax.grid(True, which='major', linestyle='--',
                    linewidth=0.6, alpha=0.7)

        plt.subplot(511)
        plt.plot(_df, label='Original')
        plt.legend(loc='upper left')
        plt.title('Série Temporal Original')
        plt.ylim(min_value, max_value)
        config_plot(plt.gca())

        plt.subplot(512)
        plt.plot(trend, label='Tendência')
        plt.legend(loc='upper left')
        plt.title('Tendência')
        config_plot(plt.gca())

        plt.subplot(513)
        plt.plot(seasonal, label='Sazonalidade')
        plt.legend(loc='upper left')
        plt.title('Sazonalidade')
        config_plot(plt.gca())

        plt.subplot(514)
        plt.plot(residual, label='Erro')
        plt.legend(loc='upper left')
        plt.title('Resíduo')
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        config_plot(plt.gca())

        plt.subplot(515)
        plt.plot(_df, label='Original', color='gray', alpha=0.5)
        plt.plot(trend, label='Tendência')
        plt.plot(seasonal, label='Sazonalidade')
        plt.plot(residual, label='Resíduo')
        plt.legend(loc='upper left')
        plt.title('Componentes sobrepostos')
        plt.ylim(min_value, max_value)
        config_plot(plt.gca())

        plt.tight_layout()
        plt.show()

        residual.dropna(inplace=True)

        self.acf_pacf_plot(residual, lags=30)
        self.plot_hist(residual)

        # Retornando os componentes
        return trend, seasonal, residual

    def check_stationarity(self, verbose=False):
        result = self.adf_test(verbose=False) and self.kpss_test(verbose=False)
        vprint(
            f"---\nA série temporal{' ' if result else ' não '}é estacionária", verbose=verbose)
        return result

    @staticmethod
    def binary_trend_accuracy(actual: pd.DataFrame, predicted: pd.DataFrame, verbose=False):
        actual_diff = actual.diff().dropna()
        predicted_diff = predicted.diff().dropna()
        trend_acc = np.mean((actual_diff >= 0) == (predicted_diff >= 0))
        actual_diff['Trend'] = actual_diff.apply(
            lambda x: 'up' if x >= 0 else 'down')
        predicted_diff['Trend'] = predicted_diff.apply(
            lambda x: 'up' if x >= 0 else 'down')
        major_class_acc = actual_diff['Trend'].value_counts(
        ).max() / actual_diff['Trend'].value_counts().sum()
        vprint(
            f'Acurácia Binária da Tendência: {trend_acc*100:.2f}% (M. Class Acc: {major_class_acc*100:.2f}%)', verbose=verbose)

        return {"acc": trend_acc, "major_class_acc": major_class_acc}

    @staticmethod
    def get_metrics(real: pd.DataFrame, pred: pd.DataFrame, verbose=False):
        mse = mean_squared_error(real, pred)
        mape = mean_absolute_percentage_error(real, pred)
        vprint(f'---\nMetrics:', verbose=verbose)
        vprint(f'Erro Quadrático Médio (MSE): {mse:.4f}', verbose=verbose)
        vprint(
            f'Erro Percentual Absoluto Médio (MAPE): {mape*100:.2f}%', verbose=verbose)
        return {"mse": mse, "mape": mape}

    def naive_model(self, train: pd.DataFrame, test: pd.DataFrame, verbose=False):
        predictions = pd.concat([train[-1:], test[:-1]])
        predictions.index = test.index

        vprint(f"---\nNaive Model:", verbose=verbose)
        metrics = self.get_metrics(test, predictions, verbose)
        btm = self.binary_trend_accuracy(test, predictions, verbose)
        vprint(f"---")

        return {"metrics": metrics, "binary_trend": btm}

    def train_test_split(self, train_size: float = 0.75, verbose=False):
        train_size = int(len(self.data) * train_size)
        train, test = self.data[:train_size], self.data[train_size:]
        vprint(
            f"---\nTamanho do conjunto de treino: {len(train)}", verbose=verbose)
        vprint(f"Tamanho do conjunto de teste: {len(test)}", verbose=verbose)
        return train, test

    def arima_model(self, order: tuple, train: pd.DataFrame, test: pd.DataFrame, auto_d=False, verbose=False):
        vprint(
            f"---\nAnalisando série temporal com ARIMA({order[0]},{order[1]},{order[2]})...", verbose=verbose)
        stationary = self.check_stationarity(verbose=verbose)
        if not stationary and not auto_d and order[1] == 0:
            vprint(
                "Considere aplicar diferenciação ou transformação.", verbose=verbose)
            return
        if auto_d:
            order = (order[0], 1, order[2])
            vprint(
                f"Aplicando diferenciação. Novo modelo ARIMA({order[0]},{order[1]},{order[2]})", verbose=verbose)

        history = [x for x in train]
        predictions = list()
        conf_int = {'max': list(), 'min': list()}

        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            forecast = model_fit.get_forecast(1)
            yhat = forecast.predicted_mean[0]
            predictions.append(yhat)
            obs = test[test.index[0]+t]
            history.append(obs)
            conf_int['max'].append(forecast.conf_int(alpha=0.05)[0][1])
            conf_int['min'].append(forecast.conf_int(alpha=0.05)[0][0])

        predictions = pd.Series(predictions, index=test.index)
        conf_int = pd.DataFrame(conf_int, index=test.index)

        metrics = self.get_metrics(test, predictions, verbose)
        bin_acc = self.binary_trend_accuracy(test, predictions, verbose)
        naive_metrics = self.naive_model(train, test, verbose)

        if verbose:
            # Plot
            plt.figure(figsize=(16, 8))
            plt.subplot(211)
            plt.plot(train, label='Treino', linestyle='-', marker='.')
            plt.plot(test, label='Teste', color='green',
                     linestyle='-', marker='.', alpha=0.5)
            plt.plot(test.index, predictions, label='Previsão',
                     color='red', linestyle='-')
            plt.fill_between(
                test.index, conf_int['min'], conf_int['max'], color='pink', alpha=0.3)
            plt.title(f'Previsão com ARIMA({order[0]},{order[1]},{order[2]})')
            plt.legend()
            # Adicionando métricas ao gráfico
            plt.text(0.01, 0.05, f'MSE: {metrics['mse']:.4f} ({naive_metrics['metrics']['mse']:.4f})\n' \
                     f'MAPE: {metrics['mape']*100:.2f}% ({naive_metrics['metrics']['mape']*100:.2f}%)\n' \
                     f'Trend: {bin_acc['acc']*100:.2f}% ({naive_metrics['binary_trend']['acc']*100:.2f}%)',
                     transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            # Ajustando a frequência das linhas de grade no eixo x
            # Define o número de ticks no eixo x
            plt.locator_params(axis='x', nbins=50)
            plt.tick_params(axis='x', which='both', direction='in', length=6,
                            width=1, labelsize=10)  # Ajusta os ticks no eixo x
            plt.xticks(rotation=45)
            plt.grid(True, which='major', linestyle='--',
                     linewidth=0.6, alpha=0.7)

            plt.subplot(212)
            plt.plot(test, label='Teste', color='green',
                     linestyle='-', marker='.', alpha=0.5)
            plt.plot(test.index, predictions, label='Previsão',
                     color='red', linestyle='-', marker='.')
            plt.fill_between(
                test.index, conf_int['min'], conf_int['max'], color='pink', alpha=0.3)
            plt.title(f'Previsões e Teste')
            plt.legend()
            # Ajustando a frequência das linhas de grade no eixo x
            # Define o número de ticks no eixo x
            plt.locator_params(axis='x', nbins=50)
            plt.tick_params(axis='x', which='both', direction='in', length=6,
                            width=1, labelsize=10)  # Ajusta os ticks no eixo x
            plt.xticks(rotation=45)
            plt.grid(True, which='major', linestyle='--',
                     linewidth=0.6, alpha=0.7)

            plt.tight_layout()
            plt.show()

        residual = pd.Series(model_fit.resid)
        ag_metric = metrics['mse'] + metrics['mape'] + \
            bin_acc['acc']  # métrica personalizada para o AG

        return metrics, bin_acc, ag_metric, residual



