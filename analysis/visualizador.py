import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DiagnosticoFlotaVisualizer:
    """
    Clase encargada de realizar el análisis exploratorio visual y estadístico unificado
    para los sensores y configuraciones de la flota de motores usando el 100% de los datos.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa la clase con los datos globales completos.
        """
        self.df = dataframe

        # Definimos de forma automática la lista de las 24 variables estándar del dataset
        self.features = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]

        # Configuramos el estilo global de Seaborn al instanciar la clase
        sns.set_theme(style="darkgrid")

    def graficar_distribucion_global(self, save_path: str = None):
        """
        Genera la cuadrícula de 24x2 con histogramas y box plots unificados
        procesando la totalidad de los datos del DataFrame.
        """
        filas = len(self.features)
        columnas = 2

        fig, axes = plt.subplots(filas, columnas, figsize=(16, filas * 3))

        print(f"Iniciando procesamiento del 100% de los datos ({len(self.df)} registros)...")
        print("Calculando histogramas y box plots para las 24 características. Por favor, espera...")

        for indice, nombre_columna in enumerate(self.features):
            if nombre_columna not in self.df.columns:
                print(f"Advertencia: La columna '{nombre_columna}' no se encuentra en el DataFrame.")
                continue

            ax_hist = axes[indice, 0]
            ax_box = axes[indice, 1]

            # Histograma
            sns.histplot(data=self.df, x=nombre_columna, kde=True, ax=ax_hist, color='teal', alpha=0.7)
            ax_hist.set_title(f'Histograma Global (Completo): {nombre_columna}', fontsize=11, fontweight='bold')
            ax_hist.set_xlabel('Valor medido', fontsize=9)
            ax_hist.set_ylabel('Conteo / Frecuencia', fontsize=9)

            # Box Plot
            sns.boxplot(data=self.df, x=nombre_columna, ax=ax_box, color='lightseagreen')
            ax_box.set_title(f'Box Plot Global (Completo): {nombre_columna}', fontsize=11, fontweight='bold')
            ax_box.set_xlabel('Valor medido', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"¡Reporte gráfico guardado exitosamente en: {save_path}!")

        plt.show()

    def obtener_tabla_estadistica(self, decimals: int = 4) -> pd.DataFrame:
        """
        Calcula y devuelve un DataFrame de Pandas con las estadísticas descriptivas
        generales y avanzadas de las 24 características basándose en el 100% de los datos.
        """
        print(f"Calculando matriz estadística detallada para {len(self.features)} variables...")

        # 1. Filtrar solo las columnas que existen en el DataFrame real para evitar errores
        columnas_validas = [col for col in self.features if col in self.df.columns]

        if not columnas_validas:
            raise ValueError("Ninguna de las 24 características estándar fue encontrada en el DataFrame.")

        # 2. Obtener estadísticas base y transponer
        tabla = self.df[columnas_validas].describe().T

        # 3. Inyectar las métricas avanzadas faltantes
        tabla['variance'] = self.df[columnas_validas].var()
        tabla['skewness'] = self.df[columnas_validas].skew()
        tabla['kurtosis'] = self.df[columnas_validas].kurt()

        # 4. Renombrar las columnas al español de forma profesional
        tabla = tabla.rename(columns={
            'count': 'Conteo',
            'mean': 'Media (Promedio)',
            'std': 'Desv. Estándar',
            'min': 'Mínimo',
            '50%': 'Mediana (Q2)',
            'max': 'Máximo',
            'variance': 'Varianza',
            'skewness': 'Sesgo (Skewness)',
            'kurtosis': 'Curtosis'
        })

        # 5. Ordenar las columnas de manera lógica para la lectura del negocio
        orden_columnas = ['Conteo', 'Media (Promedio)', 'Mediana (Q2)', 'Desv. Estándar', 'Varianza', 'Sesgo (Skewness)', 'Curtosis', 'Mínimo', 'Máximo']
        tabla_final = tabla[orden_columnas]

        # Retorna el DataFrame redondeado listo para mostrarse en el notebook
        return tabla_final.round(decimals)
