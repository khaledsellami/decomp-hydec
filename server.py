import io
import logging
import os
import shutil
from concurrent import futures
from typing import List, Union

import grpc
import numpy as np

from hydec import generate_decomposition
from utils import load_dataframe
from dataHandler import save_data
from models.hydec_pb2 import DecompRequest, Decomposition, DecompositionLayers, DecompositionLayer, \
    ApproachVersion, Granularity, Partition, AnalysisType
from models.hydec_pb2_grpc import add_HyDecServicer_to_server, HyDecServicer


TEMP_PATH = os.path.join(os.curdir, "temp_data")


def parse_layer(layer: Union[np.ndarray, List[int]], names: List[str]) -> List[Partition]:
    layer = np.array(layer)
    partitions = [Partition(name="partition{}".format(p), classes=[names[i] for i in np.where(layer==p)[0]])
                  for p in np.sort(np.unique(layer))]
    return partitions


class DecompServer(HyDecServicer):

    def getDecomposition(self, request: DecompRequest, context):
        app = request.appName
        app_repo = request.appRepo
        level = Granularity.Name(request.level)
        language = request.language
        decomp_approach = ApproachVersion.Name(request.decompApproach)
        hyperparams_path = request.hyperparams_path if request.hyperparams_path != "" else None
        structural_path = request.structural_path if request.structural_path != "" else None
        semantic_path = request.semantic_path if request.semantic_path != "" else None
        dynamic_path = request.dynamic_path if request.dynamic_path != "" else None
        final_layer, _, atoms, _ = generate_decomposition(app, app_repo, decomp_approach, hyperparams_path,
                                                                 structural_path, semantic_path, dynamic_path,
                                                                 include_metadata=False, save_output=False)
        return Decomposition(name=app, appName=app, language=language, level=level, appRepo=app_repo,
                             partitions=parse_layer(final_layer, atoms))

    def getLayers(self, request, context):
        app = request.appName
        app_repo = request.appRepo
        level = Granularity.Name(request.level)
        language = request.language
        decomp_approach = ApproachVersion.Name(request.decompApproach)
        hyperparams_path = request.hyperparams_path if request.hyperparams_path != "" else None
        structural_path = request.structural_path if request.structural_path != "" else None
        semantic_path = request.semantic_path if request.semantic_path != "" else None
        dynamic_path = request.dynamic_path if request.dynamic_path != "" else None
        final_layer, layers, atoms, _ = generate_decomposition(app, app_repo, decomp_approach, hyperparams_path,
                                                                 structural_path, semantic_path, dynamic_path,
                                                                 include_metadata=False, save_output=False)
        decomposition = Decomposition(name=app, appName=app, language=language, level=level, appRepo=app_repo,
                             partitions=parse_layer(final_layer, atoms))
        decomp_layers = [DecompositionLayer(name="layer_{}".format(i),
                                            decomposition=list(layers[i])) for i in range(len(layers))]
        layer_return = DecompositionLayers(names=atoms, layers=decomp_layers, final_decomposition=decomposition)
        return layer_return

    def getDecompositionWithFile(self, request_iterator, context):
        return self.handleRequestWithFile(request_iterator, context, self.getDecomposition)

    def getLayersWithFile(self, request_iterator, context):
        return self.handleRequestWithFile(request_iterator, context, self.getLayers)

    def handleRequestWithFile(self, request_iterator, context, function):
        bytes_data = bytearray()
        for request in request_iterator:
            one_of = request.WhichOneof('request')
            if one_of == "metadata":
                metadata = request.metadata
                file_format = metadata.format
                analysis_type = metadata.analysisType
                if analysis_type is None:
                    analysis_type = AnalysisType.DYNAMIC
            else:
                bytes_data += bytearray(request.file)
        df = load_dataframe(io.BytesIO(bytes_data), format=file_format)
        data_path = None
        try:
            data_path = save_data(df, TEMP_PATH, AnalysisType.Name(analysis_type).lower())
            if analysis_type == AnalysisType.STRUCTURAL:
                metadata.decomp_request.structural_path = data_path
            elif analysis_type == AnalysisType.DYNAMIC:
                metadata.decomp_request.dynamic_path = data_path
            elif analysis_type == AnalysisType.SEMANTIC:
                metadata.decomp_request.semantic_path = data_path
            return function(metadata.decomp_request, context)
        finally:
            if data_path is not None:
                shutil.rmtree(data_path)


def serve():
    hydec_port = os.getenv('SERVICE_HYDEC_PORT', 50055)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_HyDecServicer_to_server(DecompServer(), server)
    server.add_insecure_port(f"[::]:{hydec_port}")
    server.start()
    logging.info(f"HyDec server started, listening on {hydec_port}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    serve()
