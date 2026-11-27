#include "artifact/reader.h"
#include "targets/inkling2_mini/impl/load/bindings.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main() {
    const char* path = std::getenv("NINFER_INKLING2_MINI_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: set NINFER_INKLING2_MINI_ARTIFACT to a native Inkling v2 artifact\n";
        return 77;
    }

    try {
        using namespace ninfer::targets::inkling2_mini::detail;
        ninfer::artifact::Reader reader(path);
        if (reader.identity().model_id != "inkling2-mini" ||
            reader.identity().weights_id != "nvfp4") {
            std::cerr << "FAIL: unexpected artifact identity\n";
            return 1;
        }
        if (reader.objects().size() != kArtifactObjectCount) {
            std::cerr << "FAIL: expected " << kArtifactObjectCount << " objects, got "
                      << reader.objects().size() << '\n';
            return 1;
        }

        ninfer::artifact::Binder binder(reader);
        auto plan = bind_artifact(binder, ninfer::artifact::TensorPlacement::ValidateOnly);
        const auto& materialization = plan.materialization;
        if (materialization.object_count != kArtifactObjectCount ||
            materialization.host_objects.size() != 5 ||
            !materialization.device_objects.empty() ||
            materialization.device_capacity_bytes[0] != 0) {
            std::cerr << "FAIL: validate-only materialization plan is inconsistent\n";
            return 1;
        }
        if (plan.bindings.text_layers.size() != kPhysicalLayers ||
            plan.bindings.text_layers[0].moe.experts.size() != kExperts) {
            std::cerr << "FAIL: text binding topology is inconsistent\n";
            return 1;
        }
        std::cout << "ok: " << materialization.object_count
                  << " native objects validated without weight upload\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
