build:
	poetry build

publish: build
	poetry publish

test-publish: build
	poetry publish --dry-run
