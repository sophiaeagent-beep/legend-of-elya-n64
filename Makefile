# Legend of Elya - N64 Homebrew ROM
# World's First LLM-powered Nintendo 64 Game
BUILD_DIR=build
include $(N64_INST)/n64.mk

src = legend_of_elya.c nano_gpt.c

# No PNG/audio assets yet - just the weights binary in filesystem/
all: legend_of_elya.z64

$(BUILD_DIR)/legend_of_elya.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/legend_of_elya.elf: $(src:%.c=$(BUILD_DIR)/%.o)

legend_of_elya.z64: N64_ROM_TITLE="Legend of Elya"
legend_of_elya.z64: $(BUILD_DIR)/legend_of_elya.dfs

# nano_gpt uses trunc.w.s which R4300 FPU doesn't implement.
# -msoft-float replaces all FP instructions with software library calls.
$(BUILD_DIR)/nano_gpt.o: nano_gpt.c
	$(CC) $(N64_CFLAGS) -msoft-float -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) legend_of_elya.z64

-include $(wildcard $(BUILD_DIR)/*.d)

.PHONY: all clean
