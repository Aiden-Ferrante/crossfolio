.PHONY: install-unit

# deploy/crossfolio-campaign.service was already versioned here — what was
# missing is that it was INSTALLED AS A COPY. A copy is a second source of
# truth: edit the repo, forget to reinstall, and the file that actually runs
# disagrees with the file under review, silently. Symlinking makes that
# unrepresentable — systemd reads through the link, so git pull +
# daemon-reload is the whole update path.
#
# Cost of the choice: moving or deleting this checkout breaks the unit. That is
# the correct failure — the repo IS the deployment.
#
# The unit is deliberately NOT enabled here. A pretraining campaign is started
# when you want one, not at boot; `systemctl --user start crossfolio-campaign`.
UNIT := crossfolio-campaign.service
UNIT_DIR := $(HOME)/.config/systemd/user
install-unit:
	@mkdir -p "$(UNIT_DIR)"
	@if [ -e "$(UNIT_DIR)/$(UNIT)" ] && [ ! -L "$(UNIT_DIR)/$(UNIT)" ]; then \
	  cp "$(UNIT_DIR)/$(UNIT)" "$(UNIT_DIR)/$(UNIT).pre-repo.bak"; \
	  echo "kept the previous non-symlink unit at $(UNIT_DIR)/$(UNIT).pre-repo.bak"; \
	fi
	ln -sfn "$(CURDIR)/deploy/$(UNIT)" "$(UNIT_DIR)/$(UNIT)"
	systemctl --user daemon-reload
	@echo "linked $(UNIT_DIR)/$(UNIT) -> $(CURDIR)/deploy/$(UNIT)"
