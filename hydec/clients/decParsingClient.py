import io
import logging
import os
from typing import List, Callable, Optional

import grpc

from ..models.parse import ParserStub, NamesRequest, Granularity, ParseRequest, Status, Format
from ..utils import load_dataframe


class DecParsingClient:
    TEMP_PATH = "./temp/"
    # PARSING_PORT = 50500
    FORMAT = Format.PARQUET

    def __init__(self, app: str, app_repo: str = "", language: str = "java", granularity: str = "class",
                 is_distributed: bool = False, parser_output_path: Optional[str] = None, *args, **kwargs):
        from decparsing import select_client
        from decparsing import DataHandler as ParsingDataHandler
        analysis_client = select_client(app, app_repo, is_distributed=is_distributed, *args, **kwargs)
        self.parser = ParsingDataHandler(analysis_client, output_path=parser_output_path)
        self.app_name = app
        self.app_repo = app_repo
        self.language = language
        self.granularity = granularity
        self.is_distributed = is_distributed

    def parse_all(self):
        self.parser.load_all(self.granularity)

    def get_names(self, granularity: Granularity = None) -> List[str]:
        logging.debug(f"getting names from decparsing module")
        granularity = self.granularity if granularity is None else granularity
        return self.parser.get_names(granularity)

    def get_calls(self):
        return self.get_matrix("calls")

    def get_interactions(self):
        return self.get_matrix("interactions")

    def get_tfidf(self):
        return self.get_matrix("tfidf")

    def get_word_counts(self):
        return self.get_matrix("word_counts")

    def get_matrix(self, data_type: str):
        return self.parser.get_data(data_type, self.granularity)[1]
