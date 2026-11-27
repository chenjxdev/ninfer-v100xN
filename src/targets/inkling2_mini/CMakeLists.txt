target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/shard.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime_tp2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/draft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/draft_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)

if(NINFER_BUILD_APPS)
  add_executable(ninfer-inkling2-mini ${PROJECT_SOURCE_DIR}/apps/inkling2_mini.cpp)
  target_include_directories(ninfer-inkling2-mini PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(ninfer-inkling2-mini PRIVATE ninfer_engine ninfer_media_decode CUDA::cudart)
  add_executable(ninfer-inkling2-mini-server ${PROJECT_SOURCE_DIR}/apps/inkling2_mini_server.cpp)
  target_include_directories(ninfer-inkling2-mini-server PRIVATE
    ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer-inkling2-mini-server PRIVATE
    ninfer_engine ninfer_media_decode CUDA::cudart Threads::Threads)
endif()
