function(add_mneme target)
    # Link proteus as a shared library for preloading.
    # TODO: Change to static linking and use linker wrapper flags for interposing.
    add_proteus(${target} FORCE_JIT_ANNOTATE_ALL)
    string(TOLOWER "${CMAKE_CXX_COMPILER}" _cxx)
    set(_looks_like_clang FALSE)
    if(_cxx MATCHES "clang|amdclang|hipcc")
      set(_looks_like_clang TRUE)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR _looks_like_clang)
      target_compile_options(${target}
        PRIVATE
          -fno-discard-value-names
          -ftrivial-auto-var-init=zero
          -gline-tables-only
      )
    endif()

    # Link the user runtime for annotations and recording scopes.
    if(TARGET mnemert)
      target_link_libraries(${target} PUBLIC mnemert)
    elseif(TARGET Mneme::mnemert)
      target_link_libraries(${target} PUBLIC Mneme::mnemert)
    endif()

endfunction()
