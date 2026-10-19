# import os
# import requests
# import pandas as pd

# class ETLTools:

#     def __init__(self):
#         pass

#     def extract_load(self, url:str, output_folder:str, format:str):
#         """
#         This tool extracts the data from the API (url) and loads it into the
#         the desired location (output_folder).

#         Args:
#             url (str): The API endpoint from which to extract data.
#             output_folder (str): The folder where the extracted data will be saved.
        
#         Returns:
#             str: A message indicating the success or failure of the operation.
#         """

#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#         output_folder = os.path.join(project_root, output_folder)

#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             data = response.json()

#             filename = os.path.join(output_folder, f"extracted_data.{format}")
#             os.makedirs(output_folder, exist_ok=True)

#             df = pd.json_normalize(data['results'])
#             if format == "csv":
#                 df.to_csv(filename, index=False)
#             elif format == "json":
#                 df.to_json(filename, orient="records", lines=True)
#             elif format == "parquet":
#                 df.to_parquet(filename, index=False)
#             else:
#                 return f"Unsupported format: {format}"

#             return f"Data successfully extracted and saved to {filename}"
#         except requests.exceptions.RequestException as e:
#             return f"Failed to extract data: {e}"

#     def transform_load_context(self, file_path: str):
#         """
#         This tool transforms the data from the specified file and loads it into the
#         desired location (output_folder).

#         Args:
#             file_path (str): The path to the file containing the data to be transformed.
#             output_folder (str): The folder where the transformed data will be saved.
#             output_format (str): The format in which to save the transformed data (csv, json, parquet).
#         Returns:
#             str: A message indicating the success or failure of the operation.
#         """

#         file_extension = os.path.splitext(file_path)[1].lower()
#         if file_extension == ".csv":
#             df = pd.read_csv(file_path)
#         elif file_extension == ".json":
#             df = pd.read_json(file_path, lines=True)
#         elif file_extension == ".parquet":
#             df = pd.read_parquet(file_path)
#         else:
#             return f"Unsupported file format: {file_extension}"

#         top_3_rows = str(df.head(3))

#         return top_3_rows

#     def execute_code(self,code:str):
#         """
#         This tool executes the provided code and returns the output.

#         Args:
#             code (str): The code to be executed.
#         Returns:
#             str: The output of the executed code or an error message if execution fails.
#         """

#         try:
#             exec(code)
#             return "Code executed successfully."
#         except Exception as e:
#             return f"Failed to execute code: {e}"

# if __name__ == "__main__":
#     obj = ETLTools()
#     # print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/", "data/extract", "csv"))
#     path = "/Users/saidulmursalinkhan/Developer/Agentic_Data_Platform/data/extract/extracted data.csv"
#     print(obj.transform_load_context(path))


import os
import re
import json
import builtins
from urllib.parse import parse_qsl

import numpy as np
import requests
import pandas as pd

class ETLTools:

    SUPPORTED_FORMATS = ("csv", "json", "parquet")

    # Builtins the generated code may use (no open, no __import__, no eval/exec)
    _ALLOWED_BUILTINS = (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "print", "True", "False", "None",
    )

    # Disk access inside LLM-generated code is rejected: saving is done by save_dataframe()
    _DISK_ACCESS = re.compile(
        r"\.to_(csv|json|parquet|excel|pickle|sql|feather|hdf)\s*\(|\bpd\.read_\w+\s*\(|\bopen\s*\(|\bos\."
    )

    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # ------------------------------------------------------------------ helpers

    def _resolve(self, path: str) -> str:
        """Relative paths are resolved from the project root. Absolute paths are kept as they are."""
        path = os.path.normpath(str(path).strip().strip("'\""))
        return os.path.join(self.project_root, path)  # join() keeps an absolute `path` unchanged

    @staticmethod
    def _clean_format(format: str) -> str:
        fmt = str(format).strip().lower().lstrip(".")
        if fmt not in ETLTools.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        return fmt

    @staticmethod
    def clean_code(response: str) -> str:
        """Pull the code out of an LLM answer, with or without ``` fences."""
        match = re.search(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
        code = match.group(1) if match else response
        return code.strip()

    @staticmethod
    def _flatten_cells(df: pd.DataFrame) -> pd.DataFrame:
        """
        Lists/dicts do not survive a CSV round-trip.
        ['programming', 'python'] -> 'programming, python'  (easy to filter with str.contains)
        other nested objects      -> JSON string
        """
        def fix(value):
            if isinstance(value, (list, tuple)):
                if all(not isinstance(v, (list, tuple, dict)) for v in value):
                    return ", ".join(map(str, value))
                return json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False, default=str)
            return value

        for col in df.columns:
            df[col] = df[col].map(fix)
        return df

    # ------------------------------------------------------------------ read / save

    def read_file(self, file_path: str) -> pd.DataFrame:
        file_path = self._resolve(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            return pd.read_csv(file_path)
        elif file_extension == ".json":
            try:
                return pd.read_json(file_path, lines=True)   # files written by save_dataframe
            except ValueError:
                return pd.read_json(file_path)               # normal JSON array
        elif file_extension == ".parquet":
            return pd.read_parquet(file_path)
        raise ValueError(f"Unsupported file format: {file_extension}")

    def save_dataframe(self, df: pd.DataFrame, output_folder: str, file_name: str, format: str) -> str:
        fmt = self._clean_format(format)
        output_folder = self._resolve(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f"{file_name}.{fmt}")

        df = self._flatten_cells(df.copy())

        if fmt == "csv":
            df.to_csv(filename, index=False, encoding="utf-8")
        elif fmt == "json":
            df.to_json(filename, orient="records", lines=True, force_ascii=False)
        elif fmt == "parquet":
            try:
                df.to_parquet(filename, index=False)
            except ImportError as e:
                raise RuntimeError("Parquet needs a parquet engine: pip install pyarrow (or use csv/json).") from e
        return filename

    # ------------------------------------------------------------------ extract + load

    def extract_load(self, url:str, output_folder:str, format:str, query_params:str = ""):
        """
        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
            format (str): The format in which to save the data (csv, json, parquet).
            query_params (str): Optional filters as a query string, e.g. "tag=programming".
        
        Returns:
            str: A message indicating the success or failure of the operation.
        """

        try:
            fmt = self._clean_format(format)
        except ValueError as e:
            return str(e)

        params = dict(parse_qsl(query_params.lstrip("?"))) if query_params else None

        try:
            response = requests.get(
                url.strip(),
                params=params,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (Data_Agent ETL)"},
            )
            response.raise_for_status()
            data = response.json()

            # APIs return different shapes:
            #   PokeAPI -> {"count": ..., "results": [ {...}, ... ]}   (dict that wraps a list)
            #   dev.to  -> [ {...}, ... ]                               (bare list)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = next(
                    (v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)),
                    [data],
                )
            else:
                return f"Failed to extract data: unexpected JSON type {type(data).__name__}"

            if not records:
                return "Failed to extract data: the API returned no records."

            df = pd.json_normalize(records)   # flattens nested dicts such as user.name
            filename = self.save_dataframe(df, output_folder, "extracted_data", fmt)

            return f"Data successfully extracted and saved to {filename} ({len(df)} rows). Columns: {list(df.columns)}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"
        except Exception as e:
            return f"Failed to extract data: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------ transform + load

    def transform_load_context(self, file_path: str):
        """
        Gives the LLM the context it needs to write the transformation:
        column names, dtypes and the first 3 rows of the file.

        Args:
            file_path (str): The path to the file containing the data to be transformed.
        Returns:
            str: Columns, dtypes and top 3 rows, or an error message for unsupported files.
        """

        try:
            df = self.read_file(file_path)
        except Exception as e:
            return f"Failed to read file: {e}"

        top_3_rows = f"Columns and dtypes:\n{df.dtypes.to_string()}\n\nTop 3 rows:\n{df.head(3).to_string()}"

        return top_3_rows

    def execute_code(self, code:str, df: pd.DataFrame = None):
        """
        This tool executes the provided code and returns the output.

        With `df` (used by transform_load_tool): the code runs on a copy of df,
        must NOT touch the disk, and must assign the final DataFrame to `result_df`.
        The method returns that DataFrame and raises on any problem, so the caller can retry.

        Without `df` (old behaviour): runs the code and returns a status string.

        Args:
            code (str): The code to be executed.
            df (pd.DataFrame): Optional DataFrame the code works on.
        Returns:
            pd.DataFrame | str: result_df, or the old status string when df is not given.
        """

        if df is None:
            try:
                exec(code, {"pd": pd, "np": np})
                return "Code executed successfully."
            except Exception as e:
                return f"Failed to execute code: {e}"

        if self._DISK_ACCESS.search(code):
            raise ValueError(
                "Do not read or write files. `df` is already loaded. "
                "Only transform it and assign the final DataFrame to `result_df`."
            )

        # pd and np are provided, so import lines are not needed
        code = "\n".join(
            line for line in code.splitlines()
            if not re.match(r"\s*(import|from)\s+\w", line)
        )

        safe_builtins = {name: getattr(builtins, name) for name in self._ALLOWED_BUILTINS}
        env = {"__builtins__": safe_builtins, "pd": pd, "np": np, "df": df.copy()}

        exec(code, env)

        result_df = env.get("result_df")
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError("The code must assign the final DataFrame to a variable named `result_df`.")
        return result_df

if __name__ == "__main__":
    obj = ETLTools()
    # print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/", "data/extract", "csv"))
    # print(obj.extract_load("https://dev.to/api/articles", "data/extract", "csv", "tag=programming"))
    path = "data/extract/extracted_data.csv"   # file name uses an underscore, no space
    print(obj.transform_load_context(path))