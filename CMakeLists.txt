cmake_minimum_required(VERSION 3.18)

project(Vanta LANGUAGES CXX)

option(VANTA_ENABLE_CUDA "Build CUDA proofs and GPU CTest cases" ON)

include(CTest)
enable_testing()

find_program(VANTA_PYTEST_EXECUTABLE NAMES pytest)
if(VANTA_PYTEST_EXECUTABLE)
    add_test(
        NAME host_tests
        COMMAND ${VANTA_PYTEST_EXECUTABLE} tests -q
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
else()
    message(WARNING "pytest not found: host_tests will not be registered")
endif()

if(VANTA_ENABLE_CUDA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)

    add_executable(reduction_proof cuda/reduction_proof.cu)
    add_executable(vector_add_proof cuda/vector_add_proof.cu)

    add_test(
        NAME reduction_correct
        COMMAND $<TARGET_FILE:reduction_proof> correct
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME reduction_faulty
        COMMAND $<TARGET_FILE:reduction_proof> faulty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME vector_add_correct
        COMMAND $<TARGET_FILE:vector_add_proof> correct
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME vector_add_faulty
        COMMAND $<TARGET_FILE:vector_add_proof> faulty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME reduction_replay
        COMMAND ${Python3_EXECUTABLE} tools/replay_failure.py
                ${CMAKE_SOURCE_DIR}/failures/reduction-tail-v0.json
                --executable $<TARGET_FILE:reduction_proof> --json
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME vector_add_replay
        COMMAND ${Python3_EXECUTABLE} tools/replay_failure.py
                ${CMAKE_SOURCE_DIR}/failures/vector_add-tail-v0.json
                --executable $<TARGET_FILE:vector_add_proof> --json
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    find_program(COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer)
    if(COMPUTE_SANITIZER_EXECUTABLE)
        add_test(
            NAME reduction_memcheck
            COMMAND ${COMPUTE_SANITIZER_EXECUTABLE} --tool memcheck
                    --error-exitcode 1 $<TARGET_FILE:reduction_proof> correct
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        add_test(
            NAME vector_add_memcheck
            COMMAND ${COMPUTE_SANITIZER_EXECUTABLE} --tool memcheck
                    --error-exitcode 1 $<TARGET_FILE:vector_add_proof> correct
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()
else()
    message(STATUS "VANTA_ENABLE_CUDA=OFF: configuring host tests only")
endif()
