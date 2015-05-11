// constants.h

#ifndef SRC_CONSTANTS_H_
#define SRC_CONSTANTS_H_

// Agent parameters
#define NUM_NODES 8
#define NUM_STATES 256
#define NUM_SENSORS 2
#define NUM_MOTORS 2
#define DETERMINISTIC true

// World parameters
#define WORLD_HEIGHT 36
#define WORLD_WIDTH 16

// Evolution parameters
// Minimum length of a duplicated/deleted genome section
#define MIN_DUP_DEL_LENGTH 15
// Maximum length of a duplicated/deleted genome section
#define MAX_DUP_DEL_LENGTH 511

#endif  // SRC_CONSTANTS_H_
