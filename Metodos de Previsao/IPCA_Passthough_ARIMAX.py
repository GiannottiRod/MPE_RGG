# Analyze the time varying ARIMAX model for the FX to IPCA passthrough

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xlwings as xw

IPCA_PATH = r'D:\Documents\Mestrado\Codigo\MPE_RGG\Metodos de Previsao'

ipca_wb = xw.Book(IPCA_PATH + r'\IPCA_aberturas.xlsx')
ipca_var = ipca_wb.sheets['Variacao'].range('A5').expand('table').options(pd.DataFrame, index=False).value
ipca_wgt = ipca_wb.sheets['Ponderacao'].range('A5').expand('table').options(pd.DataFrame, index=False).value

ipca_var_T = ipca_var.drop('Descrição', axis=1).set_index('Código').T
ipca_wgt_T = ipca_wgt.drop('Descrição', axis=1).set_index('Código').T

ipca_var_T.index = pd.to_datetime(ipca_var_T.index)
ipca_wgt_T.index = pd.to_datetime(ipca_wgt_T.index)

ipca_var_T = ipca_var_T.replace('-', '0.0').astype(float) / 100
ipca_wgt_T = ipca_wgt_T.replace('-', '0.0').astype(float) / 100

group_0_cods = [0.0]
group_1_cods = [x for x in ipca_var_T.columns if 0 < x < 11]
group_2_cods = [x for x in ipca_var_T.columns if 10 < x < 101]
group_3_cods = [x for x in ipca_var_T.columns if 1000 < x < 10001]
group_4_cods = [x for x in ipca_var_T.columns if 10000 < x]

ipca_var_T_g0 = ipca_var_T[group_0_cods]
ipca_var_T_g1 = ipca_var_T[group_1_cods]
ipca_var_T_g2 = ipca_var_T[group_2_cods]
ipca_var_T_g3 = ipca_var_T[group_3_cods]
ipca_var_T_g4 = ipca_var_T[group_4_cods]

ipca_logvar_T_g0 = np.log(1 + ipca_var_T_g0)
ipca_logvar_T_g1 = np.log(1 + ipca_var_T_g1)
ipca_logvar_T_g2 = np.log(1 + ipca_var_T_g2)
ipca_logvar_T_g3 = np.log(1 + ipca_var_T_g3)
ipca_logvar_T_g4 = np.log(1 + ipca_var_T_g4)

UC1_wb = xw.Book(IPCA_PATH + r'\UC1.xlsx')
UC1 = UC1_wb.sheets[0].range('B3').expand('table').options(pd.DataFrame, index=True).value / 1000

UC1_log_var = np.log(UC1.resample('MS').last()).diff().reindex(ipca_logvar_T_g0.index)

EfeitosData = xw.Book(IPCA_PATH + r'\EfeitosData.xlsx')
EfDU = EfeitosData.sheets[0].range('B3').expand('table').options(pd.DataFrame, index=True).value


start_analysis = '2000-01-01'
end_analysis = '2023-01-01'


# for endog in [ipca_logvar_T_g0, ipca_logvar_T_g1, ipca_logvar_T_g2, ipca_logvar_T_g3, ipca_logvar_T_g4]:
for endog in [ipca_logvar_T_g0, ipca_logvar_T_g1]:
    model = ARIMA(
        endog=endog.loc[start_analysis:end_analysis],
        order=(13, 0, 0),
        exog=UC1_log_var.loc[start_analysis:end_analysis],
    )
    model_fitted = model.fit()

    print(model_fitted.summary())