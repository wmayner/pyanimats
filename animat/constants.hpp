// constants.h

#ifndef SRC_CONSTANTS_H_
#define SRC_CONSTANTS_H_


// Debug flag (comment-out to disable debugging output)
/* #define _DEBUG */

// Agent parameters
// ----------------
#define NUM_NODES 8
#define NUM_STATES (1 << NUM_NODES)
#define NUM_SENSORS 2
#define NUM_MOTORS 2
#define DETERMINISTIC true

// World parameters
// ----------------
#define WORLD_HEIGHT 36
// IMPORTANT: Must be a power of 2
#define WORLD_WIDTH 16

// Evolution parameters
// ---------------------
// IMPORTANT: These must be a power of 2
// Minimum length of a duplicated/deleted genome section
#define MIN_DUP_DEL_LENGTH 15
// Maximum length of a duplicated/deleted genome section
#define MAX_DUP_DEL_LENGTH 511

// Enumeration constants
#define WRONG_AVOID 0
#define WRONG_CATCH 1
#define CORRECT_AVOID 2
#define CORRECT_CATCH 3


#endif  // SRC_CONSTANTS_H_
