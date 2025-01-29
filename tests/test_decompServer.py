import io
import json
import os
import pickle
import unittest
import dill

import grpc
import numpy as np
import pandas as pd

from hydec.models.hydec_pb2 import DecompRequest, DecompFileRequest, Granularity, ApproachVersion, Decomposition, MetaData, \
    Format, AnalysisType
from hydec.models.hydec_pb2_grpc import HyDecStub


TEST_APP = "petclinic-legacy"


def convert(data: pd.DataFrame = None, format: Format = Format.PARQUET):
    if format == Format.PARQUET:
        return io.BytesIO(data.to_parquet())
    elif format == Format.CSV:
        return io.BytesIO(data.to_csv().encode(encoding="utf-8"))
    elif format == Format.PICKLE:
        return io.BytesIO(pickle.dumps(data))
    elif format == Format.JSON:
        return io.BytesIO(data.to_json().encode(encoding="utf-8"))
    else:
        raise ValueError("Unrecognized data_format {}!".format(format))


class TestDecompServer(unittest.TestCase):
    def test_getDecomposition(self):
        # Arrange
        test_path = os.path.join(os.curdir, "tests_data", TEST_APP)
        with open(os.path.join(test_path, "decomposition.pickle"), "rb") as f:
            expected_decomposition = dill.load(f)
        request = DecompRequest(appName=TEST_APP, language="java", level=Granularity.CLASS,
                                appRepo="whatever", decompApproach=ApproachVersion.hyDec)
        hydec_port = os.getenv('SERVICE_HYDEC_PORT', 50055)
        with grpc.insecure_channel(f'localhost:{hydec_port}') as channel:
            stub = HyDecStub(channel)
            # Act
            decomposition = stub.getDecomposition(request)
            decomposition = [partition.classes for partition in decomposition.partitions]
        # Assert
        self.assertDecompositionEqual(decomposition, expected_decomposition)
        # self.assertEqual(len(decomposition.partitions), len(expected_decomposition))
        # for partition in decomposition.partitions:
        #     exists = False
        #     partition = set(partition.classes)
        #     for expected_partition in expected_decomposition:
        #         if partition == set(expected_partition):
        #             exists = True
        #             break
        #     self.assertTrue(exists, "not all partitions match")

    def test_getLayers(self):
        # Arrange
        test_path = os.path.join(os.curdir, "tests_data", TEST_APP)
        with open(os.path.join(test_path, "layers.pickle"), "rb") as f:
            expected_layers = dill.load(f)
        with open(os.path.join(test_path, "decomposition.pickle"), "rb") as f:
            expected_decomposition = dill.load(f)
        request = DecompRequest(appName=TEST_APP, language="java", level=Granularity.CLASS,
                                appRepo="whatever", decompApproach=ApproachVersion.hyDec)
        hydec_port = os.getenv('SERVICE_HYDEC_PORT', 50055)
        with grpc.insecure_channel(f'localhost:{hydec_port}') as channel:
            stub = HyDecStub(channel)
            # Act
            response = stub.getLayers(request)
            decomposition = [partition.classes for partition in response.final_decomposition.partitions]
            layers = response.layers
            atoms = response.names
            layers_partitions = [
                [[atoms[i] for i in np.where(layer.decomposition == p)[0]] for p in np.sort(np.unique(layer.decomposition))] for layer in layers]
        # Assert
        self.assertDecompositionEqual(decomposition, expected_decomposition)
        self.assertEqual(len(layers_partitions), len(expected_layers))
        for layer, expected_layer in zip(layers_partitions, expected_layers):
            self.assertDecompositionEqual(layer, expected_layer)

    def assertDecompositionEqual(self, decomposition, expected_decomposition):
        self.assertEqual(len(decomposition), len(expected_decomposition))
        for partition in decomposition:
            exists = False
            partition = set(partition)
            for expected_partition in expected_decomposition:
                if partition == set(expected_partition):
                    exists = True
                    break
            self.assertTrue(exists, "not all partitions match")

    def test_getDecompositionWithFile(self):
        # Arrange
        test_path = os.path.join(os.curdir, "tests_data", TEST_APP)
        with open(os.path.join(test_path, "decomposition.pickle"), "rb") as f:
            expected_decomposition = dill.load(f)
        features = np.load(os.path.join(os.path.join(os.pardir, "data", TEST_APP, "dynamic", "class_calls.npy")))
        with open(os.path.join(os.pardir, "data", TEST_APP, "dynamic", "class_names.json"), "r") as f:
            atoms = json.load(f)
        df = pd.DataFrame(features, index=atoms, columns=atoms)
        data_chunks = convert(df, Format.PARQUET)
        decomp_request = DecompRequest(appName=f"{TEST_APP}", language="java", level=Granularity.CLASS,
                                appRepo="https://github.com/spring-petclinic/spring-framework-petclinic",
                                decompApproach=ApproachVersion.hyDec)
        metadata = MetaData(decomp_request=decomp_request, name=TEST_APP, format=Format.PARQUET, column_index=0,
                            row_index=0, analysisType=AnalysisType.DYNAMIC)

        def __upload_file_iterator(chunk_size=1024):
            request = DecompFileRequest(metadata=metadata)
            yield request
            with data_chunks as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    request = DecompFileRequest(file=chunk)
                    yield request

        hydec_port = os.getenv('SERVICE_HYDEC_PORT', 50055)
        with grpc.insecure_channel(f'localhost:{hydec_port}') as channel:
            stub = HyDecStub(channel)
            # Act
            decomposition = stub.getDecompositionWithFile(__upload_file_iterator())
            decomposition = [partition.classes for partition in decomposition.partitions]
        # Assert
        self.assertDecompositionEqual(decomposition, expected_decomposition)

    def test_getLayersWithFile(self):
        # Arrange
        test_path = os.path.join(os.curdir, "tests_data", TEST_APP)
        with open(os.path.join(test_path, "layers.pickle"), "rb") as f:
            expected_layers = dill.load(f)
        with open(os.path.join(test_path, "decomposition.pickle"), "rb") as f:
            expected_decomposition = dill.load(f)
        features = np.load(os.path.join(os.path.join(os.pardir, "data", TEST_APP, "dynamic", "class_calls.npy")))
        with open(os.path.join(os.pardir, "data", TEST_APP, "dynamic", "class_names.json"), "r") as f:
            atoms = json.load(f)
        df = pd.DataFrame(features, index=atoms, columns=atoms)
        data_chunks = convert(df, Format.PARQUET)
        decomp_request = DecompRequest(appName=f"{TEST_APP}", language="java", level=Granularity.CLASS,
                                appRepo="https://github.com/spring-petclinic/spring-framework-petclinic",
                                decompApproach=ApproachVersion.hyDec)
        metadata = MetaData(decomp_request=decomp_request, name=TEST_APP, format=Format.PARQUET, column_index=0,
                            row_index=0, analysisType=AnalysisType.DYNAMIC)

        def __upload_file_iterator(chunk_size=1024):
            request = DecompFileRequest(metadata=metadata)
            yield request
            with data_chunks as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    request = DecompFileRequest(file=chunk)
                    yield request

        hydec_port = os.getenv('SERVICE_HYDEC_PORT', 50055)
        with grpc.insecure_channel(f'localhost:{hydec_port}') as channel:
            stub = HyDecStub(channel)
            # Act
            response = stub.getLayersWithFile(__upload_file_iterator())
            decomposition = [partition.classes for partition in response.final_decomposition.partitions]
            layers = response.layers
            atoms = response.names
            layers_partitions = [
                [[atoms[i] for i in np.where(layer.decomposition == p)[0]] for p in np.sort(np.unique(layer.decomposition))] for layer in layers]
        # Assert
        self.assertDecompositionEqual(decomposition, expected_decomposition)
        self.assertEqual(len(layers_partitions), len(expected_layers))
        for layer, expected_layer in zip(layers_partitions, expected_layers):
            self.assertDecompositionEqual(layer, expected_layer)
