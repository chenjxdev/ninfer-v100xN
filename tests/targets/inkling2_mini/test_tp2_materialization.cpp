#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "targets/inkling2_mini/impl/load/bindings.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

int main() {
    const char* path = std::getenv("NINFER_INKLING2_MINI_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: set NINFER_INKLING2_MINI_ARTIFACT for the TP2 real-artifact test\n";
        return 77;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
        std::cout << "SKIP: two CUDA devices are required\n";
        return 77;
    }
    try {
        using namespace ninfer;
        using namespace ninfer::targets::inkling2_mini::detail;
        artifact::Reader reader(path);
        ExecutionContext execution(std::vector<int>{0, 1});
        artifact::Binder binder(reader, 2);
        auto plan = bind_artifact(binder, artifact::TensorPlacement::Device);
        for (int rank = 0; rank < 2; ++rank) {
            std::cout << "planned rank " << rank << ": "
                      << plan.materialization.device_capacity_bytes[rank] << " bytes\n";
        }
        auto materialized = artifact::materialize(reader, plan.materialization, execution);
        for (int rank = 0; rank < 2; ++rank) { execution.dev[rank]->synchronize(); }
        const auto& stats = materialized.stats();
        for (int rank = 0; rank < 2; ++rank) {
            if (stats.per_device_capacity_bytes[rank] !=
                plan.materialization.device_capacity_bytes[rank]) {
                std::cerr << "FAIL: rank capacity differs from its plan\n";
                return 1;
            }
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::cout << "materialized rank " << rank << ": "
                      << stats.per_device_capacity_bytes[rank] << " bytes, free " << free_bytes
                      << " / " << total_bytes << "\n";
        }
        std::cout << "ok: TP2 uploaded " << stats.tensor_count << " text tensors and "
                  << stats.resource_count << " resources\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
