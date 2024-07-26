import pandas as pd
import numpy as np
from datetime import date 


ir_eur_curve_6m = pd.DataFrame(np.array([['CASH', date(2024, 4, 30), date(2024, 5, 6), '3BD', 6, 0.039313111],
                                ['CASH', date(2024, 4, 30), date(2024, 5, 7), '1W', 7, 0.039336813],
                                ['CASH', date(2024, 4, 30), date(2024, 5, 14),
                                    '2W', 14, 0.039370406],
                                ['CASH', date(2024, 4, 30), date(2024, 5, 21),
                                    '3W', 21, 0.039386373],
                                ['CASH', date(2024, 4, 30), date(2024, 5, 30),
                                    '1M', 30, 0.039361],
                                ['CASH', date(2024, 4, 30), date(2024, 6, 28),
                                    '2M', 59, 0.038995],
                                ['CASH', date(2024, 4, 30), date(2024, 7, 30),
                                    '3M', 91, 0.038744],
                                ['CASH', date(2024, 4, 30), date(2024, 8, 30),
                                    '4M', 122, 0.038624],
                                ['CASH', date(2024, 4, 30), date(2024, 9, 30),
                                    '5M', 153, 0.038431],
                                ['CASH', date(2024, 4, 30), date(2024, 10, 30),
                                    '6M', 183, 0.03813],
                                ['FRA', date(2024, 5, 30), date(2024, 12, 30),
                                    'FRA-1M-7M', 244, 0.037456114],
                                ['FRA', date(2024, 6, 28), date(2025, 2, 28),
                                    'FRA-2M-8M', 304, 0.036747956],
                                ['FRA', date(2024, 7, 30), date(2025, 4, 30),
                                    'FRA-3M-9M', 365, 0.036029687],
                                ['FRA', date(2024, 8, 30), date(2025, 6, 30),
                                    'FRA-4M-10M', 426, 0.035404603],
                                ['FRA', date(2024, 9, 30), date(2025, 8, 29),
                                    'FRA-5M-11M', 486, 0.03477227],
                                ['FRA', date(2024, 10, 30), date(2025, 10, 30),
                                    'FRA-6M-12M', 548, 0.034175261],
                                ['FRA', date(2024, 11, 29), date(2025, 12, 29),
                                    'FRA-7M-13M', 608, 0.033631846],
                                ['FRA', date(2024, 12, 30), date(2026, 2, 27),
                                    'FRA-8M-14M', 668, 0.033087195],
                                ['FRA', date(2025, 1, 30), date(2026, 4, 30),
                                    'FRA-9M-15M', 730, 0.032565272],
                                ['FRA', date(2025, 2, 28), date(2026, 6, 29),
                                    'FRA-10M-16M', 790, 0.032101166],
                                ['FRA', date(2025, 3, 31), date(2026, 8, 31),
                                    'FRA-11M-17M', 853, 0.031615746],
                                ['FRA', date(2025, 4, 30), date(2026, 10, 30),
                                    'FRA-12M-18M', 913, 0.031139454],
                                ['FRA', date(2025, 5, 30), date(2026, 12, 30),
                                    'FRA-13M-19M', 974, 0.030782286],
                                ['FRA', date(2025, 6, 30), date(2027, 2, 26),
                                    'FRA-14M-20M', 1032, 0.030476446],
                                ['FRA', date(2025, 7, 30), date(2027, 4, 30),
                                    'FRA-15M-21M', 1095, 0.030112955],
                                ['FRA', date(2025, 8, 29), date(2027, 6, 29),
                                    'FRA-16M-22M', 1155, 0.0297541],
                                ['FRA', date(2025, 9, 30), date(2027, 8, 30),
                                    'FRA-17M-23M', 1217, 0.029378001],
                                ['FRA', date(2025, 10, 30), date(2027, 10, 29),
                                    'FRA-18M-24M', 1277, 0.029061516],
                                ['FRA', date(2026, 4, 30), date(2028, 10, 30),
                                    'FRA-24M-30M', 1644, 0.027628463],
                                ['SWAP', date(2024, 4, 30), date(2027, 4, 30),
                                    '3Y', 1095, 0.03188],
                                ['SWAP', date(2024, 4, 30), date(2028, 4, 28),
                                    '4Y', 1459, 0.03054],
                                ['SWAP', date(2024, 4, 30), date(2029, 4, 30),
                                    '5Y', 1826, 0.02968],
                                ['SWAP', date(2024, 4, 30), date(2030, 4, 30),
                                    '6Y', 2191, 0.02918],
                                ['SWAP', date(2024, 4, 30), date(2031, 4, 30),
                                    '7Y', 2556, 0.0289],
                                ['SWAP', date(2024, 4, 30), date(2032, 4, 30),
                                    '8Y', 2922, 0.02875],
                                ['SWAP', date(2024, 4, 30), date(2033, 4, 29),
                                    '9Y', 3286, 0.02869],
                                ['SWAP', date(2024, 4, 30), date(2034, 4, 28),
                                    '10Y', 3650, 0.02869],
                                ['SWAP', date(2024, 4, 30), date(2035, 4, 30),
                                    '11Y', 4017, 0.028709554],
                                ['SWAP', date(2024, 4, 30), date(2036, 4, 30),
                                    '12Y', 4383, 0.02874],
                                ['SWAP', date(2024, 4, 30), date(2037, 4, 30),
                                    '13Y', 4748, 0.028758942],
                                ['SWAP', date(2024, 4, 30), date(2038, 4, 30),
                                    '14Y', 5113, 0.028755258],
                                ['SWAP', date(2024, 4, 30), date(2039, 4, 29),
                                    '15Y', 5477, 0.02871],
                                ['SWAP', date(2024, 4, 30), date(2040, 4, 30),
                                    '16Y', 5844, 0.028613427],
                                ['SWAP', date(2024, 4, 30), date(2041, 4, 30),
                                    '17Y', 6209, 0.028464075],
                                ['SWAP', date(2024, 4, 30), date(2042, 4, 30),
                                    '18Y', 6574, 0.02827543],
                                ['SWAP', date(2024, 4, 30), date(2043, 4, 30),
                                    '19Y', 6939, 0.02806049],
                                ['SWAP', date(2024, 4, 30), date(2044, 4, 29),
                                    '20Y', 7304, 0.02783],
                                ['SWAP', date(2024, 4, 30), date(2045, 4, 28),
                                    '21Y', 7668, 0.027587082],
                                ['SWAP', date(2024, 4, 30), date(2046, 4, 30),
                                    '22Y', 8035, 0.027338756],
                                ['SWAP', date(2024, 4, 30), date(2047, 4, 30),
                                    '23Y', 8400, 0.027087761],
                                ['SWAP', date(2024, 4, 30), date(2048, 4, 30),
                                    '24Y', 8766, 0.026837677],
                                ['SWAP', date(2024, 4, 30), date(2049, 4, 30),
                                    '25Y', 9131, 0.02659],
                                ['SWAP', date(2024, 4, 30), date(2050, 4, 29),
                                    '26Y', 9495, 0.026346703],
                                ['SWAP', date(2024, 4, 30), date(2051, 4, 28),
                                    '27Y', 9859, 0.026109874],
                                ['SWAP', date(2024, 4, 30), date(2052, 4, 30),
                                    '28Y', 10227, 0.02588184],
                                ['SWAP', date(2024, 4, 30), date(2053, 4, 30),
                                    '29Y', 10592, 0.025660098],
                                ['SWAP', date(2024, 4, 30), date(2054, 4, 30),
                                    '30Y', 10957, 0.02545],
                                ['SWAP', date(2024, 4, 30), date(2055, 4, 30),
                                    '31Y', 11322, 0.025250244],
                                ['SWAP', date(2024, 4, 30), date(2056, 4, 28),
                                    '32Y', 11686, 0.025059878],
                                ['SWAP', date(2024, 4, 30), date(2057, 4, 30),
                                    '33Y', 12053, 0.024875626],
                                ['SWAP', date(2024, 4, 30), date(2058, 4, 30),
                                    '34Y', 12418, 0.024696605],
                                ['SWAP', date(2024, 4, 30), date(2059, 4, 30),
                                    '35Y', 12783, 0.02452],
                                ['SWAP', date(2024, 4, 30), date(2060, 4, 30),
                                    '36Y', 13149, 0.02434789],
                                ['SWAP', date(2024, 4, 30), date(2061, 4, 29),
                                    '37Y', 13513, 0.024175195],
                                ['SWAP', date(2024, 4, 30), date(2062, 4, 28),
                                    '38Y', 13877, 0.024003297],
                                ['SWAP', date(2024, 4, 30), date(2063, 4, 30),
                                    '39Y', 14244, 0.023831818],
                                ['SWAP', date(2024, 4, 30), date(2064, 4, 30),
                                    '40Y', 14610, 0.02366],
                                ['SWAP', date(2024, 4, 30), date(2074, 4, 30),
                                    '50Y', 18262, 0.02208],
                                ['SWAP', date(2024, 4, 30), date(2084, 4, 28),
                                    '60Y', 21913, 0.02108]], dtype=object))
ir_eur_curve_6m.columns = ['type', 'start', 'maturity', 'tenor', 'daycount', 'quote']

ir_eur_curve_estr = pd.DataFrame(np.array([['CASH', date(2024, 4, 30), date(2024, 5, 2),
                            '1BD', 2, 0.03906],
                        ['CASH', date(2024, 4, 30), date(2024, 5, 3),
                            '2BD', 3, 0.03906],
                        ['CASH', date(2024, 4, 30), date(2024, 5, 6),
                            '3BD', 6, 0.039072261],
                        ['SWAP', date(2024, 4, 30), date(2024, 5, 7),
                            '1W', 7, 0.03909],
                        ['SWAP', date(2024, 4, 30), date(2024, 5, 14),
                            '2W', 14, 0.03911],
                        ['SWAP', date(2024, 4, 30), date(2024, 5, 21),
                            '3W', 21, 0.03912],
                        ['SWAP', date(2024, 4, 30), date(2024, 5, 30),
                            '1M', 30, 0.03912],
                        ['SWAP', date(2024, 4, 30), date(2024, 6, 28),
                            '2M', 59, 0.03841],
                        ['SWAP', date(2024, 4, 30), date(2024, 7, 30),
                            '3M', 91, 0.03785],
                        ['SWAP', date(2024, 4, 30), date(2024, 8, 30),
                            '4M', 122, 0.03752],
                        ['SWAP', date(2024, 4, 30), date(2024, 9, 30),
                            '5M', 153, 0.03717],
                        ['SWAP', date(2024, 4, 30), date(2024, 10, 30),
                            '6M', 183, 0.03677],
                        ['SWAP', date(2024, 4, 30), date(2024, 11, 29),
                            '7M', 213, 0.03646],
                        ['SWAP', date(2024, 4, 30), date(2024, 12, 30),
                            '8M', 244, 0.03615],
                        ['SWAP', date(2024, 4, 30), date(2025, 1, 30),
                            '9M', 275, 0.03582],
                        ['SWAP', date(2024, 4, 30), date(2025, 2, 28),
                            '10M', 304, 0.03554],
                        ['SWAP', date(2024, 4, 30), date(2025, 3, 31),
                            '11M', 335, 0.03525],
                        ['SWAP', date(2024, 4, 30), date(2025, 4, 30),
                            '1Y', 365, 0.03494],
                        ['SWAP', date(2024, 4, 30), date(2025, 5, 30),
                            '13M', 395, 0.0345828],
                        ['SWAP', date(2024, 4, 30), date(2025, 6, 30),
                            '14M', 426, 0.034256265],
                        ['SWAP', date(2024, 4, 30), date(2025, 7, 30),
                            '15M', 456, 0.03393],
                        ['SWAP', date(2024, 4, 30), date(2025, 8, 29),
                            '16M', 486, 0.03363838],
                        ['SWAP', date(2024, 4, 30), date(2025, 9, 30),
                            '17M', 518, 0.033356217],
                        ['SWAP', date(2024, 4, 30), date(2025, 10, 30),
                            '18M', 548, 0.03308],
                        ['SWAP', date(2024, 4, 30), date(2025, 11, 28),
                            '19M', 577, 0.03283361],
                        ['SWAP', date(2024, 4, 30), date(2025, 12, 30),
                            '20M', 609, 0.032581264],
                        ['SWAP', date(2024, 4, 30), date(2026, 1, 30),
                            '21M', 640, 0.03237],
                        ['SWAP', date(2024, 4, 30), date(2026, 2, 27),
                            '22M', 668, 0.032174968],
                        ['SWAP', date(2024, 4, 30), date(2026, 3, 30),
                            '23M', 699, 0.031941762],
                        ['SWAP', date(2024, 4, 30), date(2026, 4, 30),
                            '2Y', 730, 0.03177],
                        ['SWAP', date(2024, 4, 30), date(2027, 4, 30),
                            '3Y', 1095, 0.02974],
                        ['SWAP', date(2024, 4, 30), date(2028, 4, 28),
                            '4Y', 1459, 0.02842],
                        ['SWAP', date(2024, 4, 30), date(2029, 4, 30),
                            '5Y', 1826, 0.02763],
                        ['SWAP', date(2024, 4, 30), date(2030, 4, 30),
                            '6Y', 2191, 0.02721],
                        ['SWAP', date(2024, 4, 30), date(2031, 4, 30),
                            '7Y', 2556, 0.02702],
                        ['SWAP', date(2024, 4, 30), date(2032, 4, 30),
                            '8Y', 2922, 0.02695],
                        ['SWAP', date(2024, 4, 30), date(2033, 4, 29),
                            '9Y', 3286, 0.02701],
                        ['SWAP', date(2024, 4, 30), date(2034, 4, 28),
                            '10Y', 3650, 0.02712],
                        ['SWAP', date(2024, 4, 30), date(2035, 4, 30),
                            '11Y', 4017, 0.027265194],
                        ['SWAP', date(2024, 4, 30), date(2036, 4, 30),
                            '12Y', 4383, 0.02743],
                        ['SWAP', date(2024, 4, 30), date(2037, 4, 30),
                            '13Y', 4748, 0.027587057],
                        ['SWAP', date(2024, 4, 30), date(2038, 4, 30),
                            '14Y', 5113, 0.02771706],
                        ['SWAP', date(2024, 4, 30), date(2039, 4, 29),
                            '15Y', 5477, 0.0278],
                        ['SWAP', date(2024, 4, 30), date(2040, 4, 30),
                            '16Y', 5844, 0.027820872],
                        ['SWAP', date(2024, 4, 30), date(2041, 4, 30),
                            '17Y', 6209, 0.027785126],
                        ['SWAP', date(2024, 4, 30), date(2042, 4, 30),
                            '18Y', 6574, 0.027702761],
                        ['SWAP', date(2024, 4, 30), date(2043, 4, 30),
                            '19Y', 6939, 0.027584793],
                        ['SWAP', date(2024, 4, 30), date(2044, 4, 29),
                            '20Y', 7304, 0.02744],
                        ['SWAP', date(2024, 4, 30), date(2045, 4, 28),
                            '21Y', 7668, 0.027277185],
                        ['SWAP', date(2024, 4, 30), date(2046, 4, 30),
                            '22Y', 8035, 0.027101559],
                        ['SWAP', date(2024, 4, 30), date(2047, 4, 30),
                            '23Y', 8400, 0.026917431],
                        ['SWAP', date(2024, 4, 30), date(2048, 4, 30),
                            '24Y', 8766, 0.026728034],
                        ['SWAP', date(2024, 4, 30), date(2049, 4, 30),
                            '25Y', 9131, 0.02654],
                        ['SWAP', date(2024, 4, 30), date(2050, 4, 29),
                            '26Y', 9495, 0.026354904],
                        ['SWAP', date(2024, 4, 30), date(2051, 4, 28),
                            '27Y', 9859, 0.026174621],
                        ['SWAP', date(2024, 4, 30), date(2052, 4, 30),
                            '28Y', 10227, 0.025999527],
                        ['SWAP', date(2024, 4, 30), date(2053, 4, 30),
                            '29Y', 10592, 0.025830481],
                        ['SWAP', date(2024, 4, 30), date(2054, 4, 30),
                            '30Y', 10957, 0.02567],
                        ['SWAP', date(2024, 4, 30), date(2059, 4, 30),
                            '35Y', 12783, 0.024965548],
                        ['SWAP', date(2024, 4, 30), date(2064, 4, 30),
                            '40Y', 14610, 0.02435],
                        ['SWAP', date(2024, 4, 30), date(2074, 4, 30),
                            '50Y', 18262, 0.02319],
                        ['SWAP', date(2024, 4, 30), date(2084, 4, 28),
                            '60Y', 21913, 0.02234]], dtype=object) )
ir_eur_curve_estr.columns = ['type', 'start', 'maturity', 'tenor', 'daycount', 'quote']
ir_eur_disc_estr = pd.DataFrame(np.array([['IR', 'EUR-ESTR-ON', 'DF-30-Apr-2024', 'MID', 1.0,
        date(2024, 4, 30), 0],
       ['IR', 'EUR-ESTR-ON', 'DF-02-May-2024', 'MID', 0.999783047,
        date(2024, 5, 2), 2],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2024', 'MID', 0.999674582,
        date(2024, 5, 3), 3],
       ['IR', 'EUR-ESTR-ON', 'DF-06-May-2024', 'MID', 0.999349192,
        date(2024, 5, 6), 6],
       ['IR', 'EUR-ESTR-ON', 'DF-10-May-2024', 'MID', 0.998915323,
        date(2024, 5, 10), 10],
       ['IR', 'EUR-ESTR-ON', 'DF-17-May-2024', 'MID', 0.998156442,
        date(2024, 5, 17), 17],
       ['IR', 'EUR-ESTR-ON', 'DF-24-May-2024', 'MID', 0.997398519,
        date(2024, 5, 24), 24],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Jun-2024', 'MID', 0.996318318,
        date(2024, 6, 3), 34],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Jul-2024', 'MID', 0.99321041,
        date(2024, 7, 3), 64],
       ['IR', 'EUR-ESTR-ON', 'DF-05-Aug-2024', 'MID', 0.98989143,
        date(2024, 8, 5), 97],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Sep-2024', 'MID', 0.987021623,
        date(2024, 9, 3), 126],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Oct-2024', 'MID', 0.984128065,
        date(2024, 10, 3), 156],
       ['IR', 'EUR-ESTR-ON', 'DF-04-Nov-2024', 'MID', 0.981135349,
        date(2024, 11, 4), 188],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Dec-2024', 'MID', 0.978467815,
        date(2024, 12, 3), 217],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Jan-2025', 'MID', 0.975671042,
        date(2025, 1, 3), 248],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Feb-2025', 'MID', 0.972955284,
        date(2025, 2, 3), 279],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Mar-2025', 'MID', 0.970546958,
        date(2025, 3, 3), 307],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Apr-2025', 'MID', 0.967924638,
        date(2025, 4, 3), 338],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2025', 'MID', 0.965291489,
        date(2025, 5, 5), 370],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Jun-2025', 'MID', 0.962943773,
        date(2025, 6, 3), 399],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Jul-2025', 'MID', 0.960547672,
        date(2025, 7, 3), 429],
       ['IR', 'EUR-ESTR-ON', 'DF-04-Aug-2025', 'MID', 0.958037255,
        date(2025, 8, 4), 461],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Sep-2025', 'MID', 0.955734744,
        date(2025, 9, 3), 491],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Oct-2025', 'MID', 0.95347708,
        date(2025, 10, 3), 521],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Nov-2025', 'MID', 0.951181142,
        date(2025, 11, 3), 552],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Dec-2025', 'MID', 0.9489857,
        date(2025, 12, 3), 582],
       ['IR', 'EUR-ESTR-ON', 'DF-05-Jan-2026', 'MID', 0.946599573,
        date(2026, 1, 5), 615],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Feb-2026', 'MID', 0.944529681,
        date(2026, 2, 3), 644],
       ['IR', 'EUR-ESTR-ON', 'DF-03-Mar-2026', 'MID', 0.942556904,
        date(2026, 3, 3), 672],
       ['IR', 'EUR-ESTR-ON', 'DF-07-Apr-2026', 'MID', 0.940124431,
        date(2026, 4, 7), 707],
       ['IR', 'EUR-ESTR-ON', 'DF-04-May-2026', 'MID', 0.938270887,
        date(2026, 5, 4), 734],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2027', 'MID', 0.914689271,
        date(2027, 5, 3), 1098],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2028', 'MID', 0.89266884,
        date(2028, 5, 3), 1464],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2029', 'MID', 0.871236706,
        date(2029, 5, 3), 1829],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2030', 'MID', 0.849746342,
        date(2030, 5, 3), 2194],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2031', 'MID', 0.827986053,
        date(2031, 5, 5), 2561],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2032', 'MID', 0.80645538,
        date(2032, 5, 3), 2925],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2033', 'MID', 0.784540458,
        date(2033, 5, 3), 3290],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2034', 'MID', 0.762692801,
        date(2034, 5, 3), 3655],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2035', 'MID', 0.740941557,
        date(2035, 5, 3), 4020],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2036', 'MID', 0.719211269,
        date(2036, 5, 5), 4388],
       ['IR', 'EUR-ESTR-ON', 'DF-04-May-2037', 'MID', 0.698132014,
        date(2037, 5, 4), 4752],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2038', 'MID', 0.677717937,
        date(2038, 5, 3), 5116],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2039', 'MID', 0.658202355,
        date(2039, 5, 3), 5481],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2040', 'MID', 0.639848156,
        date(2040, 5, 3), 5847],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2041', 'MID', 0.622766515,
        date(2041, 5, 3), 6212],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2042', 'MID', 0.606748365,
        date(2042, 5, 5), 6579],
       ['IR', 'EUR-ESTR-ON', 'DF-04-May-2043', 'MID', 0.591912381,
        date(2043, 5, 4), 6943],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2044', 'MID', 0.577972883,
        date(2044, 5, 3), 7308],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2045', 'MID', 0.564853436,
        date(2045, 5, 3), 7673],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2046', 'MID', 0.552472256,
        date(2046, 5, 3), 8038],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2047', 'MID', 0.540752713,
        date(2047, 5, 3), 8403],
       ['IR', 'EUR-ESTR-ON', 'DF-04-May-2048', 'MID', 0.529552659,
        date(2048, 5, 4), 8770],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2049', 'MID', 0.518934465,
        date(2049, 5, 3), 9134],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2050', 'MID', 0.508694464,
        date(2050, 5, 3), 9499],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2051', 'MID', 0.498815406,
        date(2051, 5, 3), 9864],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2052', 'MID', 0.489234043,
        date(2052, 5, 3), 10230],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2053', 'MID', 0.479915375,
        date(2053, 5, 5), 10597],
       ['IR', 'EUR-ESTR-ON', 'DF-04-May-2054', 'MID', 0.470921686,
        date(2054, 5, 4), 10961],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2059', 'MID', 0.429007799,
        date(2059, 5, 5), 12788],
       ['IR', 'EUR-ESTR-ON', 'DF-05-May-2064', 'MID', 0.392812354,
        date(2064, 5, 5), 14615],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2074', 'MID', 0.336863422,
        date(2074, 5, 3), 18265],
       ['IR', 'EUR-ESTR-ON', 'DF-03-May-2084', 'MID', 0.290704602,
        date(2084, 5, 3), 21918]], dtype=object))
ir_eur_disc_estr.columns = ['type', 'name', 'curve_instrument', 'quote_type', 'market_quote', 'maturity_date', 'daycount']